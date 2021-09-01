import subprocess
import os
import time
logs_folder = "logs"
if not os.path.exists(logs_folder):
  os.mkdir(logs_folder)

config_1 = "testpat1_back"
config_2 = "testpat1_fsi"
archivo_salida1 =  open(os.path.join(logs_folder, "log_"+config_1+".txt"), "w")
archivo_salida2 =  open(os.path.join(logs_folder, "log_"+config_2+".txt"), "w")
archivo_error1 =  open(os.path.join(logs_folder, "err_"+config_1+".txt"), "w")
archivo_error2 =  open(os.path.join(logs_folder, "err_"+config_2+".txt"), "w")
#subprocess.Popen(["./main_prdip.py", "config_files/config_cameraman_fsi.json"]) 
p1 = subprocess.Popen(["python", "main_prdip.py", "config_files/config _"+config_1+".json"], stdout=archivo_salida1, stderr=archivo_error1, bufsize=1)
p2 = subprocess.Popen(["python", "main_prdip.py", "config_files/config _"+config_2+".json"], stdout=archivo_salida2, stderr=archivo_error2, bufsize=1)
processes = [p1,p2]
while True:
    
    if all(p.poll() is None for p in processes):
        print("process running.")
        time.sleep(3)
        print("process running..")
        time.sleep(3)
        print("process running...")
        time.sleep(3)
    else:
        print("running process check out log and error files to follow the results")
        break
