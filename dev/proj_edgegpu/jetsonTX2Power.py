# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
import threading as thd
import time

class PowerReadingValue(object):
	def __init__(self, name, path):
		self.name = name
		self.path = path
		self.value = ""

	def update(self):
		with open(self.path, 'r') as myfile:
			self.value=myfile.read().rstrip()

	def __str__(self):
		return "{0} : {1}".format(self.path, self.value)

	def get_name(self):
		return self.name

	def get_value(self):
		return self.value

class PowerReadingPower(PowerReadingValue):
	def __init__(self, path, number):
		PowerReadingValue.__init__(self, "power", path+"/in_power{0}_input".format(number))

class PowerReadingRail(object):
	def __init__(self, path, num):
		with open(path+"/rail_name_{0}".format(num), 'r') as myfile:
			self.name = myfile.read().rstrip()
			self.power = PowerReadingPower(path, num)

	def __str__(self):
		s = self.name + "\n"
		s += "\t" + str(self.power) + "\n"
		return s

	def get_name(self):
		return self.name

	def update(self):
		self.power.update()

	def to_csv(self):
		d=OrderedDict()
		d[self.name+' '+self.power.get_name()]=self.power.get_value()
		return d

	def get_csv_header(self):
		return [self.name+' '+self.power.get_name()]

class PowerReadingDevice(object):
	def __init__(self, path):
		self.rails = []
		for i in range(1):
			self.rails.append(PowerReadingRail(path, i))

	def __str__(self):
		s=""
		for r in self.rails:
			s += str(r)
		return s

	def update(self):
		for r in self.rails:
			r.update()

	def get_csv_header(self):
		s=[]
		for r in self.rails:
			s.extend(r.get_csv_header())
		return s

	def to_csv(self):
		s=OrderedDict()
		for r in self.rails:
			s.update(r.to_csv())
		return s

def print_power(devices):
	for d in devices:
		d.update()
		row = d.to_csv()
		print(row["VDD_IN power"], type(row["VDD_IN power"]))

def create_devices():
	devices = []

	paths= ["/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device",]
	for p in paths:
		devices.append(PowerReadingDevice(p))

	return devices

def record_energy(devices, energy_data_list):
	d = devices[0]
	d.update()
	row = d.to_csv()
	v = int(row["VDD_IN power"])
	v = float(v)
	energy_data_list.append(v)

class EnergyMonitor(object):
	def __init__(self, T=0.1):
		self.pwr_devices = create_devices()
		self.energy_data_list = []
		self.T = T

	def reset_energy_data_list(self):
		self.energy_data_list = []

	def start(self):
		record_energy(self.pwr_devices, self.energy_data_list)
		self.monitor_thd = thd.Timer(self.T ,self.start)
		self.monitor_thd.daemon = True # stop if the program exits
		self.monitor_thd.start()

	def stop(self): # stop monitor thread
 		self.monitor_thd.cancel()

def main():
	monitor = EnergyMonitor()
	monitor.reset_energy_data_list()
	monitor.start()
	time.sleep(1) # process
	monitor.stop()
	time.sleep(2)
	print(monitor.energy_data_list, len(monitor.energy_data_list))

if __name__ == "__main__":
	main()
