
from msilib import CAB
import time
import clr
import pandas as pd
import threading
import logging
import random



clr.AddReference('EngineIO')

from EngineIO import *
'''
print("Home I/O & Python are such good friends thanks to pythonnet!")

livingRoomLight = MemoryMap.Instance.GetBit(0, MemoryType.Output)

for i in range(5):
    livingRoomLight.Value = not livingRoomLight.Value

    # When using a memory value before calling the Update method we are using a cached value.
    print("Light is on? " + str(livingRoomLight.Value))

    # Calling the Update method will write the livingRoomLight.Value to the memory map.
    MemoryMap.Instance.Update()

    time.sleep(1)

# When we no longer need the MemoryMap we should call the Dispose method to release all the allocated resources.
MemoryMap.Instance.Dispose()

print("Bye!")
'''

def thread_f(name):
    logging.info("Thread %s: starting",name)
    time_start = 7 * 3600
    time_end = 8 * 3600
    
    random_seconds = random.randint(time_start,time_end)
    random_timestamp = '{:02d}:{:02d}:{:02d}'.format(random_seconds // 3600, (random_seconds // 60) % 60, random_seconds % 60)
    
    print("Time started: ",random_timestamp)
    logging.info("Thread %s: finishing", name)
    
    #creates when the actvity between a set timeframe, with the hour betwen 0 and 24, and the minutes between 1 and 60
def timestamp_start(start_hour, start_min, end_hour, end_min):
    time_start = start_hour * 3600 + start_min * 60
    time_end = end_hour * 3600 + end_min * 60
    
    random_seconds = random.randint(time_start,time_end)
    random_timestamp = '{:02d}:{:02d}:{:02d}'.format(random_seconds // 3600, (random_seconds // 60) % 60, random_seconds % 60)
    
    return random_timestamp

#Creating empty dataset
data=pd.DataFrame(columns = ['timestamp', 'duration(s)', 'activity', 'sensor_data'])

user_input=-1

while user_input!=0:
    print("which dataset section do you want to make?\n\nOption 1) Breakfast\nOption 2) Lunch\nOption 3) Dinner\nOption 4) Personal Hygiene\nOption 5) Bathroom Break\n"
          "Option 6) Sleeping\nOption 7) Relaxing\nOption 8) Watching TV\nOption 9) Exercise\nOption 10) Work\nOption 11) Socializing\nOption 12) Reading\nOption "
          "13) Cooking\nOption 14) Gardening\nOption 15) Cleaning\nOption 16) Laundry\nOption 17) Hobbies\n\nOption 0) Finish\n")

    user_input = int(input("Please enter a number: \n"))



    match user_input:
        case 0: 
            data.to_csv('adl_dataset.csv', index=False)   
            print("Dataset saved to csv file")
        case 1: 
            print("Generating Breakfast Data...\n...\n...\n\n")
            for i in range(20):

                timestamp = timestamp_start(7,30,8,40)
                time_lapses=[]
                sensor_data=""
    
                #making cereal 
                cabinet_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"cabinet_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                cabinet_door.Value = not cabinet_door.Value; sensor_data=sensor_data+"cabinet_door:OFF"+", "
                time_lapses.append(random.randint(15,21)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                fridge_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"fridge_door:ON"+", "
                time_lapses.append(random.randint(68,82)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                fridge_door.Value = not fridge_door.Value; sensor_data=sensor_data+"fridge_door:OFF"+", "
                time_lapses.append(random.randint(4,8)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                kitchen_draw = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"kitchen_draw:ON"+", "
                time_lapses.append(random.randint(4,8)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                kitchen_draw.Value = not cabinet_door.Value; sensor_data=sensor_data+"kitchen_draw:OFF"+", "
                time_lapses.append(random.randint(4,8)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                time_lapses.append(random.randint(150,340)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair.Value = not cabinet_door.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="breakfast_cereal"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Breakfast Data Generated\n\n")
        case 6:
            print("Generating Sleeping Data...\n...\n...\n\n")
            for i in range(20):
                
                timestamp = timestamp_start(20,45,22,0)
                time_lapses=[]
                sensor_data=""
                
                #sleeping
                bed_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bed_sensor:ON"+", "
                time_lapses.append(random.randint(6*3600,9*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                bed_sensor.Value = not bed_sensor.Value; sensor_data=sensor_data+"bed_sensor:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="sleeping"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Sleeping Data Generated\n\n")

            #bfast=threading.Thread(target=thread_f,args=(1,))
