from abc import ABCMeta, abstractclassmethod
from multiprocessing import Process
import collections
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


SARD = collections.namedtuple('SARD', ['state', 'action', 'reward', 'done'])


class Client(Process):
    """Env是Clinet，向Agent发送动作请求。"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    def run(self):
        context = zmq.Context()
        require_socket = context.socket(zmq.REQ)
        push_socket = context.socket(zmq.PUSH)
        reply_socket = context.socket(zmq.REP)

        while True:

            message = self._get_reply()
            require_socket.send(msgpack.packb(message))

            message = require_socket.recv()
            action = msgpack.unpackb(message)

            self._perform_action(action)

            if self._is_done() is True:
                trajectory = self._get_trajectory()
                push_socket.send(msgpack.packb(trajectory))

                # 等待Master Agent的唤醒。同步处理。
                msg = reply_socket.recv()
                print("msg: {}".format(msg))
                reply_socket.send(b"I get awake.")

        require_socket.close()
        push_socket.close()
        reply_socket.close()
        context.term()

    @abstractclassmethod
    def newgame(self):
        """初始化游戏。"""

    @abstractclassmethod
    def endgame(self):
        """结束游戏。"""

    @abstractclassmethod
    def _perform_action(self, action):
        """运行来自Agent的动作action；收集trajectory信息。"""

    @abstractclassmethod
    def _get_reply(self):
        """整理发送给Agent的消息。"""

    @abstractclassmethod
    def _is_done(self):
        """判断游戏是否结束。"""

    @abstractclassmethod
    def _get_trajectory(self):
        """返回收集的trajectory。"""


class Worker(Process):
    """Agent是Worker，处理来自Env发过来的信息，回复动作。"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    def run(self):
        context = zmq.Context()
        reply_socket = zmq.socket(zmq.REP)
        sub_socket = zmq.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        while True:

            if self._is_wait() is True:
                model = sub_socket.recv()
                self._update_policy(model)

            message = reply_socket.recv()
            message = msgpack.unpackb(message)

            action = self._get_action(message)
            reply_socket.send(msgpack.packb(action))

        reply_socket.close()
        sub_socket.close()
        context.term()

    @abstractclassmethod
    def initialize(self):
        """初始化。设置合适的model。"""

    @abstractclassmethod
    def _get_action(self, message):
        """计算处理接受信息的动作；处理是否等待更新。"""

    @abstractclassmethod
    def _is_wait(self):
        """是否等待游戏的到来。"""

    @abstractclassmethod
    def _update_policy(self, model):
        """更新Policy。"""


class Broker(Process):
    """中间代理人。将消息发送给空闲的Client或者Worker。"""

    def __init__(self):
        super().__init__()

    def run(self):
        context = zmq.Context()
        router_socket = context.socket(zmq.ROUTER)
        dealer_socket = context.socket(zmq.DEALER)

        zmq.proxy(router_socket, dealer_socket)

        router_socket.close()
        dealer_socket.close()
        context.term()


class Master(Process):
    """Master Agent。训练和分发Policy；启动整个过程的入口。"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.n_worker = 4

    def run(self):
        context = zmq.Context()
        pub_socket = context.socket(zmq.PUB)
        pull_socket = context.socket(zmq.PULL)

        while True:
            model = self._get_model()
            pub_socket.send(msgpack.packb(model))

            for _ in range(self.n_worker):
                trajectory = pull_socket.recv()
                self._collect_trajectory(trajectory)

            self._update_model()

    @abstractclassmethod
    def initialize(self):
        """初始化。"""

    @abstractclassmethod
    def _get_model(self):
        """获取分发的模型。"""

    @abstractclassmethod
    def _update_model(self):
        """跟新模型。"""

    @abstractclassmethod
    def _collect_trajectory(self):
        """收集trajectory信息，为update policy做准备。"""
