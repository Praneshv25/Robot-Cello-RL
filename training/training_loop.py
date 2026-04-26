import numpy as np
import threading
import queue
from models.gp_gate import GPGate
from models.policy import Policy
from reward.reward import Reward
from rl.sac import SAC

import dashboard_client
import rtde_control
import rtde_receive
import rtde_io as rt_io

import time
import numpy as np
import pandas as pd
import mido
from mido import MidiFile

from Baseline_Runners.rtde_single_note_servoL import *
from ..variables import *

def _build_state(tcp: np.ndarray, ft: np.ndarray, speed: float,
                 accel: float, bow_pos: float, force: float) -> np.ndarray:
    """
    Builds the 16-dim state vector from raw RTDE readings.

    tcp     — 6DoF TCP position [x, y, z, rx, ry, rz]
    ft      — 6DoF force/torque [fx, fy, fz, tx, ty, tz]
    speed   — scalar magnitude of TCP velocity
    accel   — scalar magnitude of TCP acceleration
    bow_pos — normalized position along bow (frog=0, tip=1)
    force   — current z-axis force being applied (N)

    Returns: np.ndarray of shape (16,)
    """
    return np.concatenate([
        tcp,                        # indices 0-5
        ft,                         # indices 6-11
        np.array([speed]),          # index 12
        np.array([accel]),          # index 13
        np.array([bow_pos]),        # index 14
        np.array([force]),          # index 15
    ]).astype(np.float32)


class TrainingLoop:
    def __init__(
        self,
        gp: GPGate,
        reward_calc: Reward,
        warmup_data: np.ndarray,
        config: dict
    ):
        """
        Orchestrates the two-thread training loop.
        Receives GP, reward calculator, and warmup data from main.py.
        SAC is instantiated internally since it owns its own policy and buffer.

        Note: policy.py is not used here — SAC owns the actor network directly.
        """

        self.gp          = gp
        self.reward_calc = reward_calc
        self.warmup_data = warmup_data  # kept for periodic GP retraining
        self.config      = config

        # --- SAC ---
        # owns actor, critics, replay buffer, and update logic.
        # GP is NOT passed to SAC — gating is handled in this file.
        self.sac = SAC(config=config)

        # --- Inter-thread queues ---
        # interval_queue: RTDE thread → RL thread
        #   carries state, wav_path, force after each interval completes
        # force_queue: RL thread → RTDE thread
        #   carries scalar z-force for next interval
        # maxsize=1 ensures threads always work with most recent value
        self.interval_queue = queue.Queue(maxsize=1)
        self.force_queue    = queue.Queue(maxsize=1)

        # --- Thread synchronization events ---
        # rtde_done_event: RTDE sets when interval complete, RL waits on it
        # rl_done_event:   RL sets when update complete, RTDE waits on it
        self.rtde_done_event = threading.Event()
        self.rl_done_event   = threading.Event()

        # --- Stop signal ---
        # set by run() when training is complete or interrupted
        # both threads check this in their while loops
        self.stop_event = threading.Event()

        # --- Training counters ---
        self.real_execution_count = 0
        self.retrain_every_n      = config["gp"]["retrain_every_n_real"]
        self.total_episodes       = config["rl"]["total_episodes"]

        # --- GP observations accumulated during training ---
        # combined with warmup_data when GP is periodically retrained
        self.gp_observations = []

        # --- GP rejection penalty ---
        # added to replay buffer when GP rejects a proposed force
        # so SAC learns to avoid unsafe forces
        self.gp_penalty = config["rl"]["gp_rejection_penalty"]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """
        Spins up both threads and blocks until training is complete
        or interrupted. Handles clean shutdown on KeyboardInterrupt.
        """
        print("Starting training loop...")

        rtde_thread = threading.Thread(
            target=self._rtde_thread,
            daemon=True,
            name="RTDEThread"
        )
        rl_thread = threading.Thread(
            target=self._rl_thread,
            daemon=True,
            name="RLThread"
        )

        rtde_thread.start()
        rl_thread.start()

        try:
            rtde_thread.join()
            rl_thread.join()
        except KeyboardInterrupt:
            print("Training interrupted. Stopping threads cleanly...")
            self.stop_event.set()
            rtde_thread.join()
            rl_thread.join()
            print("Threads stopped.")

        print("Training complete.")

    # ------------------------------------------------------------------
    # RTDE thread
    # ------------------------------------------------------------------

    def _rtde_thread(self):
        """
        Continuously executes force intervals on the robot.
        Holds at interval boundaries until RL thread signals it is done.
        Never increments iteration count while waiting for RL.
        """
        current_force        = self.config["force"]["initial"]
        iterations_per_interval = self.config["robot"]["iterations_per_interval"]  # 100
        intervals_per_episode   = self.config["robot"]["intervals_per_note"]        # 5
        episode_count = 0

        while not self.stop_event.is_set():
            for interval in range(intervals_per_episode):

                # --- Execute 100 iterations at current force ---
                # PLACEHOLDER: replace with actual RTDE call
                # should send current_force along z-axis for
                # iterations_per_interval cycles
                force = current_force

                # --- Collect state components from RTDE ---
                # PLACEHOLDER: replace with actual RTDE reads
                tcp     = rtde_r.getActualTCPPose()        # (6,) array
                ft      = rtde_r.getActualForce()         # (6,) array
                speed   = spd      # scalar
                accel   = acceleration     # scalar
                bow_pos = _get_bow_pos_placeholder()    # scalar 0-1

                # PLACEHOLDER: get wav path from audio recording script
                wav_path = _get_wav_path_placeholder()

                # build 16-dim state from RTDE readings
                state = _build_state(tcp, ft, speed, accel, bow_pos, current_force)

                # --- Signal RL thread that interval is complete ---
                self.interval_queue.put({
                    "state":    state,
                    "wav_path": wav_path,
                    "force":    current_force,
                    "interval": interval,
                    "tcp":      tcp,
                    "ft":       ft,
                })
                self.rtde_done_event.set()

                # --- Hold here until RL thread is done ---
                # bow continues to physically move and string continues
                # to vibrate at current_force, but iteration count
                # does not advance until new force is received
                self.rl_done_event.wait()
                self.rl_done_event.clear()

                # --- Pick up new force from RL thread ---
                try:
                    current_force = self.force_queue.get_nowait()
                except queue.Empty:
                    # should not happen given event synchronization
                    # but guard against it anyway
                    print("Warning: no new force in queue, keeping current force.")

            # --- Episode complete ---
            episode_count += 1
            print(f"Episode {episode_count}/{self.total_episodes} complete.")

            if episode_count >= self.total_episodes:
                print("Total episodes reached. Stopping RTDE thread.")
                self.stop_event.set()
                break

    # ------------------------------------------------------------------
    # RL thread
    # ------------------------------------------------------------------

    def _rl_thread(self):
        """
        Waits for RTDE thread to complete each interval.
        Scores audio, computes reward, updates SAC, gates via GP,
        and puts new force in queue for RTDE thread.
        """
        prev_state = None  # tracks previous state for next_obs in SAC store()

        while not self.stop_event.is_set():

            # --- Wait for RTDE thread to finish an interval ---
            # blocks here while RTDE is executing 100 iterations
            self.rtde_done_event.wait()
            self.rtde_done_event.clear()

            # --- Unpack interval data ---
            try:
                data = self.interval_queue.get_nowait()
            except queue.Empty:
                print("Warning: interval queue empty. Skipping update.")
                self.rl_done_event.set()
                continue

            state    = data["state"]
            wav_path = data["wav_path"]
            force    = data["force"]
            interval = data["interval"]
            tcp      = data["tcp"]
            ft       = data["ft"]

            # --- Compute reward ---
            # reward_calc passes .wav to CNN classifier and converts
            # raw score to RL reward (currently score * 1.0)
            reward = self.reward_calc.compute_reward(wav_path)

            # --- Store transition in SAC replay buffer ---
            # done=True at end of episode (last interval), False otherwise
            done = (interval == self.config["robot"]["intervals_per_note"] - 1)

            if prev_state is not None:
                # store previous transition now that we have next_state
                self.sac.store(
                    obs=prev_state,
                    force=force,
                    reward=reward,
                    next_obs=state,
                    done=done
                )

            # --- SAC update ---
            # Step 1: update actor/critic networks from replay buffer
            # returns empty dict if buffer too small
            losses = self.sac.update()
            if losses:
                print(f"Interval {interval} | "
                      f"critic_loss: {losses['critic_loss']:.4f} | "
                      f"actor_loss: {losses['actor_loss']:.4f} | "
                      f"alpha: {losses['alpha']:.4f}")

            # Step 2: query updated actor for proposed force
            proposed_force = self.sac.select_action(state)

            # Step 3: GP gate — check if proposed force is safe
            # GP gates on (force, tcp) — extracts these from state
            if self.gp.is_approved(
                force=np.array([proposed_force]),
                tcp=tcp
            ):
                # approved — send to RTDE thread
                self.force_queue.put(proposed_force)
            else:
                # rejected — add penalty transition so SAC learns
                # to avoid this force, keep current force
                print(f"Warning: GP rejected proposed force {proposed_force:.3f}N. "
                      f"Keeping current force {force:.3f}N.")
                self.sac.store(
                    obs=state,
                    force=proposed_force,
                    reward=self.gp_penalty,   # large negative reward
                    next_obs=state,           # next_obs same as obs since not executed
                    done=False
                )
                self.force_queue.put(force)   # keep current force

            # --- Periodic GP retraining ---
            self.real_execution_count += 1
            if self.real_execution_count % self.retrain_every_n == 0:
                print(f"Retraining GP on {self.real_execution_count} "
                      f"real observations...")
                self.gp.add_observation(
                    force=np.array([force]),
                    tcp=tcp
                )
                self.gp.fit()

            # --- Update prev_state for next iteration ---
            prev_state = state

            # --- Signal RTDE thread that RL is done ---
            # RTDE thread is holding at interval boundary waiting for this
            self.rl_done_event.set()
