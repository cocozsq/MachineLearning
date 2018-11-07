import queue as Q
import heapq as H
que = Q.PriorityQueue()

heap = [2,3,4,-8,1,0,2,2,2,4]
# que.put(heapq.push(4))
# H.heappush(heap,2)
# H.heappush(heap,-8)
# H.heappush(heap,18)
# H.heappush(heap,8)
# print(H.nsmallest(3,heap))
print(H.nsmallest(3,heap))
k_neighbors = H.nsmallest(5,heap)
print(k_neighbors)
# que.put(2)
# que.put(-8)
# que.put(18)
# que.put(8)
# print(que.qsize())
# while not que.empty():
# 	print(que.get())

arrays = [1,2,3,4,5,6]
arr = indexable(arrays)
print(arr)
