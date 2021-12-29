class MyList(list):
    def __init__(self, *args):
        super(MyList, self).__init__(args)

    def __sub__(self, other):
        return self.__class__(*[item for item in self if item not in other])

# def subtract_list(list1, list2):
#     z = MyList(list1) - MyList(list2)
#     return z

