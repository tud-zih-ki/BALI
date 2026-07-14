import logging
import multiprocessing as mp
import time
import pynvml


class GPUSamplingHandler():
    def __init__(self, sampling_method: str = "nvml", sampling_interval: int = 0.1):

        self._start_event = mp.Event()
        self._stop_event = mp.Event()
        self._result_queue = mp.Queue()
        self._terminate_event = mp.Event()
        self._process = mp.Process(
            target=self._worker_loop,
            args=(
                self._start_event,
                self._stop_event,
                self._terminate_event,
                self._result_queue,
                sampling_method,
                sampling_interval
            ),
        )

    def spawn(self):
        """Start the worker process (it stays idle)."""
        self._process.start()

    def start_measurement(self):
        """Trigger the worker to start a new session of work."""
        self._start_event.set()

    def stop_measurement(self):
        """Stop the current session and return its result."""
        self._stop_event.set()
        result = self._result_queue.get()
        # reset for next iteration
        self._start_event.clear()
        self._stop_event.clear()
        return result

    def close(self):
        """Terminate the worker process gracefully."""
        self._terminate_event.set()
        # wake up if it's waiting on start
        self._start_event.set()
        self._process.join()

    @staticmethod
    def _worker_loop(start_event, stop_event, terminate_event, result_queue, sampling_method, sampling_interval):
        sampler = None
        if sampling_method == "nvml":
            sampler = NvmlSampler()
        else:
            raise NotImplementedError(f"{sampling_method} is not implemented for sampling, yet")

        logging.info("GPU sampling worker initialized with %s", sampling_method)

        while not terminate_event.is_set():
            start_event.wait()
            if terminate_event.is_set():
                break

            logging.info("GPU sampling started")
            result = []
            while not stop_event.wait(timeout=sampling_interval):
                # TODO accurate sampling interval
                result.append({
                    "time" : time.time(),
                    "memory_used" : sampler.get_used_memory(),
                    "memory_util" : sampler.get_mem_util(),
                    "gpu_util" : sampler.get_gpu_util(),
                    "power" : sampler.get_power(),
                    "temperatur" : sampler.get_temperature()
                })

            result_queue.put(result)
            logging.info("GPU sampling stopped")

            start_event.clear()
            stop_event.clear()

        logging.info("GPU sampling shutdown")

class GpuSampler:
    def __init__(self, gpu_idx):
        pass

    def get_used_memory(self, gpu_idx):
        pass

    def get_mem_util(self):
        pass

    def get_gpu_util(self, gpu_idx):
        pass

    def get_temperature(self, gpu_idx):
        pass

    def get_power(self, gpu_idx):
        pass

    def __del__(self):
        pass

class NvmlSampler(GpuSampler):
    def __init__(self, gpu_idx: int = None):
        logging.info("Initialising GPU sampling with NVML")

        pynvml.nvmlInit()

        self.gpu_handle = {}
        if gpu_idx is None:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                self.gpu_handle[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
        else:
            self.gpu_handle[0] = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)


    def get_used_memory(self, gpu_idx: int = 0):
        return pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle[gpu_idx]).used


    def get_mem_util(self, gpu_idx: int = 0):
        return pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle[gpu_idx]).memory


    def get_gpu_util(self, gpu_idx: int = 0):
        return pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle[gpu_idx]).gpu


    def get_temperature(self, gpu_idx: int = 0):
        return pynvml.nvmlDeviceGetTemperatureV(
            self.gpu_handle[gpu_idx], pynvml.NVML_TEMPERATURE_GPU)


    def get_power(self, gpu_idx: int = 0):
        return pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle[gpu_idx])


    def __del__(self):
        pynvml.nvmlShutdown()

class RocmSampler(GpuSampler):
    def __init__(self, gpu_idx=None):
        pass

    def get_used_memory(self, gpu_idx=0):
        pass

    def get_mem_util(self):
        pass

    def get_gpu_util(self, gpu_idx=0):
        pass

    def get_temperature(self, gpu_idx=0):
        pass

    def get_power(self, gpu_idx=0):
        pass

    def __del__(self):
        pass
