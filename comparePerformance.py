import subprocess
import psutil
import time
import os


if __name__ == "__main__":
	# TIMING
	p1StartTime = time.time()
	# /TIMING
	p1 = subprocess.Popen('"../venv/Scripts/activate.bat" && python stl.py formula.stl signals/results/angles_ac_cart-pole.csv quantitative', cwd=f"{os.getcwd()}/stl2", shell=True)
	# MEMORY
	# p1MemLog = []
	# process1 = psutil.Process(p1.pid)
	# try:
	# 	while p1.poll() is None:
	# 		p1MemLog.append(process1.memory_info().rss)
	# 		time.sleep(0.5)
	# except BaseException as e:
	# 	print(f"Caught exception: {e}")
	# 	pass
	# / MEMORY
	# TIMING
	while p1.poll() is None:
		pass
	p1EndTime = time.time()
	print(f"OldImpl time taken: {p1EndTime - p1StartTime}")
	# / TIMING
	# TIMING
	p2StartTime = time.time()
	# /TIMING
	p2 = subprocess.Popen('"../venv/Scripts/activate.bat" && python main.py formula.stl ../stl2/signals/results/angles_ac_cart-pole.csv quantitative', cwd=f"{os.getcwd()}/stlTool", shell=True)
	# MEMORY
	# process2 = psutil.Process(p2.pid)
	# p2MemLog = []
	# try:
	# 	while p2.poll() is None:
	# 		p2MemLog.append(process2.memory_info().rss)
	# 		time.sleep(0.5)
	# except:
	# 	pass
	# / MEMORY
	# TIMING
	while p2.poll() is None:
		pass

	p2EndTime = time.time()
	print(f"NewImpl time taken: {p2EndTime - p2StartTime}")
	# / TIMING

	# MEMORY WRITE
	# We store the results in files because we can't run efficient and syntax algorithms
	# one after the other. It requires a modification (or calling it differently - by importing package)
	# So, instead, we write pickled files with the data, then visualize it using plot.py
	# import pickle
	# with open("p1memlog.pickle", "rb") as f:
	# 	pickle.dump(p1MemLog, f, pickle.HIGHEST_PROTOCOL)
	# with open("p2memlog.pickle", "rb") as f:
	# 	pickle.dump(p2MemLog, f, pickle.HIGHEST_PROTOCOL)
	# with open("p2memlogeff.pickle", "rb") as f:
	# 	pickle.dump(p2MemLog, f, pickle.HIGHEST_PROTOCOL)

	# This will only work when the memory segments are uncommented.
	# When using the efficient algorithm (enabled in UntilNode), the p2memlogeff should be used
	# where 'eff' indicates use of the efficient algorithm.

	# Sample result visualization
	# import matplotlib.pyplot as plt
	# plt.plot(p1MemLog, color='blue', label='Pre-existing implementation memory use')
	# plt.plot(p2MemLog, color='red', label='New implementation memory use')
	# plt.title("Memory use for syntax algorithm comparison")
	# plt.legend()
	# plt.show()