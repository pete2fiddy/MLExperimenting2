
class Queue:

    def __init__(self):
        self.__elems = []

    def push(self, data):
        self.__elems.append(data)

    def pop(self):
        to_return = self.__elems[len(self.__elems)-1]
        del self.__elems[len(self.__elems)-1]
        return to_return

    def isempty(self):
        return len(self.__elems) == 0
