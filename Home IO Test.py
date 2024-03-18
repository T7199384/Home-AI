
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