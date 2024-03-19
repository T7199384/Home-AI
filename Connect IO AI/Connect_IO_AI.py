
from msilib import CAB
from os import read
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

    #creates when the actvity between a set timeframe but doesn't fit in with cooking time
def timestamp_start_avoid_cooking(start_hour, start_min, end_hour, end_min):
    
    time_start = start_hour * 3600 + start_min * 60
    time_end = end_hour * 3600 + end_min * 60
    
    random_seconds = random.randint(time_start,time_end)
    while random_seconds >= 23400 and random_seconds <= 31200 or random_seconds >= 39000 and random_seconds <= 45000 or random_seconds >= 63000 and random_seconds <= 70200:
        random_seconds = random.randint(time_start,time_end)
    random_timestamp = '{:02d}:{:02d}:{:02d}'.format(random_seconds // 3600, (random_seconds // 60) % 60, random_seconds % 60)
    
    return random_timestamp

#Creating empty dataset
data=pd.DataFrame(columns = ['timestamp', 'duration(s)', 'activity', 'sensor_data'])

user_input=-1
all_flag=False

while user_input!=0:
    if all_flag==False:
        print("which dataset section do you want to make?\n\nOption 1) Breakfast\nOption 2) Lunch\nOption 3) Dinner\nOption 4) Personal Hygiene\nOption 5) Bathroom Break\n"
              "Option 6) Sleeping\nOption 7) Relaxing\nOption 8) Watching TV\nOption 9) Exercise\nOption 10) Work\nOption 11) Socializing\nOption 12) Reading\nOption "
              "13) Cooking\nOption 14) Gardening\nOption 15) Cleaning\nOption 16) Laundry\nOption 17) Dishes\n\nOption 18) All\n\nOption 0) Finish\n")

    if user_input!=18 and all_flag==False:
        user_input = int(input("Please enter a number: \n"))
    else:
        all_flag=True
        user_input=user_input-1



    match user_input:
        case 0: 
            data.to_csv('adl_dataset.csv', index=False)   
            print("Dataset saved to csv file")
        case 1: 
            print("Generating Breakfast Data...\n...\n...\n\n")
            for i in range(200):

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
                kitchen_draw.Value = not kitchen_draw.Value; sensor_data=sensor_data+"kitchen_draw:OFF"+", "
                time_lapses.append(random.randint(4,8)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                time_lapses.append(random.randint(150,340)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair.Value = not dining_chair.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="breakfast_cereal"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Breakfast Data Generated\n\n")
        case 2: 
            print("Generating Lunch Data...\n...\n...\n\n")
            for i in range(200):

                timestamp = timestamp_start(11,50,12,30)
                time_lapses=[]
                sensor_data=""
    
                #making sandwich 
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
                kitchen_draw.Value = not kitchen_draw.Value; sensor_data=sensor_data+"kitchen_draw:OFF"+", "
                time_lapses.append(random.randint(5*60,7*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                time_lapses.append(random.randint(150,340)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair.Value = not dining_chair.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="lunch_sandwich"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Lunch Data Generated\n\n")
        case 3: 
            print("Generating Dinner Data...\n...\n...\n\n")
            for i in range(200):

                timestamp = timestamp_start(18,30,19,30)
                time_lapses=[]
                sensor_data=""
    
                #making pasta 
                oven_hob = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"oven_hob:ON"+", "
                time_lapses.append(random.randint(12,15)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                oven_hob.Value = not oven_hob.Value; sensor_data=sensor_data+"oven_hob:OFF"+", "
                
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
                kitchen_draw.Value = not kitchen_draw.Value; sensor_data=sensor_data+"kitchen_draw:OFF"+", "
                time_lapses.append(random.randint(5*60,7*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                time_lapses.append(random.randint(150,340)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair.Value = not dining_chair.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="dinner_pasta"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Dinner Data Generated\n\n")
        case 4:
            print("Generating Hygiene Data...\n...\n...\n\n")
            for i in range(200):

                timestamp = timestamp_start(8,30,20,0)
                time_lapses=[]
                sensor_data=""
    
                #personal hygiene
                bathroom_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bathroom_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                bathroom_door.Value = not bathroom_door.Value; sensor_data=sensor_data+"bathroom_door:OFF"+", "
                shower_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"shower_sensor:ON"+", "
                time_lapses.append(random.randint(5,20*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                shower_sensor.Value = not shower_sensor.Value; sensor_data=sensor_data+"shower_sensor:OFF"+", "
                bathroom_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bathroom_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                bathroom_door.Value = not bathroom_door.Value; sensor_data=sensor_data+"bathroom_door:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="personal_hygiene"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Hygiene Data Generated\n\n")
        case 5:
            print("Generating Bathroom Data...\n...\n...\n\n")
            for i in range(200):

                timestamp = timestamp_start(8,30,22,0)
                time_lapses=[]
                sensor_data=""
    
                #bathroom break
                bathroom_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bathroom_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                bathroom_door.Value = not bathroom_door.Value; sensor_data=sensor_data+"bathroom_door:OFF"+", "
                
                time_lapses.append(random.randint(45,15*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "

                bathroom_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bathroom_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                bathroom_door.Value = not bathroom_door.Value; sensor_data=sensor_data+"bathroom_door:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="bathroom_break"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Bathroom Data Generated\n\n")
        case 6:
            print("Generating Sleeping Data...\n...\n...\n\n")
            for i in range(200):
                
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
        case 7:
            print("Generating Relaxing Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp = timestamp_start(10,0,20,0)
                time_lapses=[]
                sensor_data=""
                
                #Relaxing
                #chance of where to relax
                relax_spot=random.randint(1,10)
                if relax_spot >= 1 and relax_spot <=3:
                    bed_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bed_sensor:ON"+", "
                    time_lapses.append(random.randint(1*1800,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                    bed_sensor.Value = not bed_sensor.Value; sensor_data=sensor_data+"bed_sensor:OFF"+", "
                elif relax_spot == 4:
                    dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                    time_lapses.append(random.randint(1*1800,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                    dining_chair.Value = not dining_chair.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                else:
                    sofa_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sofa_sensor:ON"+", "
                    time_lapses.append(random.randint(1*1800,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                    sofa_sensor.Value = not sofa_sensor.Value; sensor_data=sensor_data+"sofa_sensor:OFF"+", "
            
                time_total=sum(time_lapses)
            
                activity="relaxing"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Relaxing Data Generated\n\n")
        case 8:
            print("Generating TV Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp_int = timestamp_start(10,0,20,0)
                time_lapses=[]
                sensor_data=""
                
                #watching TV
                sofa_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sofa_sensor:ON"+", "
                time_lapses.append(random.randint(10,40)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                tv_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"tv_sensor:ON"+", "
                time_lapses.append(random.randint(1*1200,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                tv_sensor.Value = not tv_sensor.Value; sensor_data=sensor_data+"tv_sensor:OFF"+", "
                time_lapses.append(random.randint(5,14)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                sofa_sensor.Value = not sofa_sensor.Value; sensor_data=sensor_data+"sofa_sensor:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="watching_tv"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("TV Data Generated\n\n")
        case 9:
            print("Generating Exercise Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp_int = timestamp_start_avoid_cooking(10,0,18,0)
                time_lapses=[]
                sensor_data=""
                
                #Using treadmill
                treadmill = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"treadmill:ON"+", "
                time_lapses.append(random.randint(1800,3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                treadmill.Value = not treadmill.Value; sensor_data=sensor_data+"treadmill:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="exercising"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("TV Exercise Generated\n\n")
        case 10:
            print("Generating Work Data...\n...\n...\n\n")
            for i in range(200):
                
                
                random_seconds = 8 * 3600 + random.randint(45,90)*60
                timestamp = '{:02d}:{:02d}:{:02d}'.format(random_seconds // 3600, (random_seconds // 60) % 60, random_seconds % 60)
                time_lapses=[]
                sensor_data=""
                
                #working
                office_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"office_chair:ON"+", "
                time_lapses.append(random.randint(10,40)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                computer_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"computer_sensor:ON"+", "
                time_lapses.append(random.randint(1*3600,8*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                computer_sensor.Value = not computer_sensor.Value; sensor_data=sensor_data+"computer_sensor:OFF"+", "
                time_lapses.append(random.randint(5,14)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                office_chair.Value = not office_chair.Value; sensor_data=sensor_data+"office_chair:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="working"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Work Data Generated\n\n")
        case 10:
            print("Generating Socialising Data...\n...\n...\n\n")
            for i in range(200):
                
                
                timestamp = timestamp_start(9,0,20,0)
                time_lapses=[]
                sensor_data=""
                
                #socialising
                phone_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"phone_sensor:ON"+", "
                time_lapses.append(random.randint(300,3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                phone_sensor.Value = not phone_sensor.Value; sensor_data=sensor_data+"phone_sensor:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="socialising"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Work Socialising Generated\n\n")
        case 12:
            print("Generating Reading Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp = timestamp_start(10,0,20,0)
                time_lapses=[]
                sensor_data=""
                
                #Reading
                bookcase = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bookcase:ON"+", "
                time_lapses.append(random.randint(10,60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                bookcase.Value = not bookcase.Value; sensor_data=sensor_data+"bookcase:OFF"+", "
                #chance of where to read
                reading_spot=random.randint(1,10)
                if reading_spot == 1:
                    bed_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"bed_sensor:ON"+", "
                    time_lapses.append(random.randint(1*1800,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                    bed_sensor.Value = not bed_sensor.Value; sensor_data=sensor_data+"bed_sensor:OFF"+", "
                elif reading_spot == 2 or reading_spot == 3:
                    dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                    time_lapses.append(random.randint(1*1800,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                    dining_chair.Value = not dining_chair.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                else:
                    sofa_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sofa_sensor:ON"+", "
                    time_lapses.append(random.randint(1*1800,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                    sofa_sensor.Value = not sofa_sensor.Value; sensor_data=sensor_data+"sofa_sensor:OFF"+", "
            
                time_total=sum(time_lapses)
            
                activity="reading"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Reading Data Generated\n\n")
        case 13: 
            print("Generating Cooking Data...\n...\n...\n\n")
            for i in range(200):

                timestamp = timestamp_start_avoid_cooking(9,30,21,0)
                time_lapses=[]
                sensor_data=""
    
                #making snack 
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
                kitchen_draw.Value = not kitchen_draw.Value; sensor_data=sensor_data+"kitchen_draw:OFF"+", "
                time_lapses.append(random.randint(4,8)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"dining_chair:ON"+", "
                time_lapses.append(random.randint(150,340)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                dining_chair.Value = not dining_chair.Value; sensor_data=sensor_data+"dining_chair:OFF"+", "
                
                time_total=sum(time_lapses)
            
                activity="making_snack"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Cooking Data Generated\n\n")
        case 14: 
            print("Generating Gardening Data...\n...\n...\n\n")
            for i in range(200):

                timestamp = timestamp_start_avoid_cooking(10,0,17,0)
                time_lapses=[]
                sensor_data=""
    
                #gardening
                front_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"front_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                front_door.Value = not front_door.Value; sensor_data=sensor_data+"front_door:OFF"+", "
                time_lapses.append(random.randint(3*60,4*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                hose_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"hose_sensor:ON"+", "
                time_lapses.append(random.randint(25*60,35*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                hose_sensor.Value = not hose_sensor.Value; sensor_data=sensor_data+"hose_sensor:OFF"+", "
                time_lapses.append(random.randint(3*60,4*60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                front_door = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"front_door:ON"+", "
                time_lapses.append(random.randint(12,24)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                front_door.Value = not front_door.Value; sensor_data=sensor_data+"front_door:OFF"+", "
                
                
                time_total=sum(time_lapses)
            
                activity="gardening"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Gardening Data Generated\n\n")
        case 15:
            print("Generating Cleaning Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp = timestamp_start(10,0,16,0)
                time_lapses=[]
                sensor_data=""
                
                #Cleaning
                sink_cabinet = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sink_cabinet:ON"+", "
                time_lapses.append(random.randint(30,60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                sink_cabinet.Value = not sink_cabinet.Value; sensor_data=sensor_data+"sink_cabinet:OFF"+", "
                time_lapses.append(random.randint(3600,2*3600)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                vaccuum_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"vaccuum_sensor:ON"+", "
                time_lapses.append(random.randint(3600,4800)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                vaccuum_sensor.Value = not vaccuum_sensor.Value; sensor_data=sensor_data+"vaccuum_sensor:OFF"+", "
                time_lapses.append(random.randint(60,180)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                sink_cabinet = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sink_cabinet:ON"+", "
                time_lapses.append(random.randint(30,60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                sink_cabinet.Value = not sink_cabinet.Value; sensor_data=sensor_data+"sink_cabinet:OFF"+", "
            
                time_total=sum(time_lapses)
            
                activity="cleaning"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Cleaning Data Generated\n\n")
        case 16:
            print("Generating Laundry Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp = timestamp_start(10,0,14,0)
                time_lapses=[]
                sensor_data=""
                
                #Laundry
                sink_cabinet = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sink_cabinet:ON"+", "
                time_lapses.append(random.randint(30,60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                sink_cabinet.Value = not sink_cabinet.Value; sensor_data=sensor_data+"sink_cabinet:OFF"+", "
                laundry_sensor = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"laundry_sensor:ON"+", "
                time_lapses.append(random.randint(60,70)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                laundry_sensor.Value = not laundry_sensor.Value; sensor_data=sensor_data+"laundry_sensor:OFF"+", "
            
                time_total=sum(time_lapses)
            
                activity="laundry"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
            print("Laundry Data Generated\n\n")
        case 17:
            print("Generating Dishes Data...\n...\n...\n\n")
            for i in range(200):
                
                timestamp = timestamp_start(18,30,19,30)
                time_lapses=[]
                sensor_data=""
                
                #washing dishes
                sink_cabinet = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"sink_cabinet:ON"+", "
                time_lapses.append(random.randint(30,60)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                sink_cabinet.Value = not sink_cabinet.Value; sensor_data=sensor_data+"sink_cabinet:OFF"+", "
                kitchen_sink = MemoryMap.Instance.GetBit(0,MemoryType.Output); sensor_data=sensor_data+"kitchen_sink:ON"+", "
                time_lapses.append(random.randint(30,50)); sensor_data=sensor_data+"t:"+str(time_lapses[-1])+", "
                kitchen_sink.Value = not kitchen_sink.Value; sensor_data=sensor_data+"kitchen_sink:OFF"+", "
            
                time_total=sum(time_lapses)
            
                activity="cleaning_dishes"
                
                data.loc[len(data)]=[timestamp,time_total,activity,sensor_data]
        case 18:
            print("Dishes Data Generated\n\n")
            #bfast=threading.Thread(target=thread_f,args=(1,))
