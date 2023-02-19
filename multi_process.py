from multiprocessing import Process, Queue
import time
  
def send_to_queue(q, mylist): 
    # function to put elements into Queue
    for num in mylist: 
        print('putting ', num)
        q.put(num)
  
def pick_from_queue(q, mylist): 
    # function to print queue elements 
    print("Queue elements:")
    t1 = time.time()
    for num in mylist:
        result = q.get()
        print('getting ', num, 'th:', result)

    print("Queue is now empty!", time.time()-t1) 
  
if __name__ == "__main__": 
    # input list 
    mylist = [i for i in range(100)] 
  
    # creating multiprocessing Queue 
    q = Queue()
  
    # creating new processes 
    p1 = Process(target=send_to_queue, args=(q, mylist)) 
    p2 = Process(target=pick_from_queue, args=(q, mylist))
  
    # running process p1 and p2
    p1.start()
    p2.start() 
  
    # if we need waiting p1 and p2 process to finish their job
    #p1.join() 
    #p2.join() 