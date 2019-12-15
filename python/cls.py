class MyClass:
    def func1(cls, a, b):
        print(cls)

    @classmethod
    def func2(cls, a, b):
        print(cls)

    @staticmethod
    def func3(cls, a, b):
        print(cls)


x = MyClass()
x.func1(1, 2)
x.func2(1, 2)
x.func3(1, 2, 3)
