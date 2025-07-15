# zmq_pub_sub
Example code for a high volume ZeroMQ based publisher and subscriber.

I use a makeshift pythonic solution to maintain a constant publishing rate and `asyncio` to create a rate agnostic subscriber.


### Run

```
python zmq_publisher.py --rate 2500 --shape 1000,1000 --dtype float32

python zmq_subscriber.py
```
