from functools import wraps

def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        print(args[0])
        return func(*args, **kwargs)
    return with_logging

@logit
def addition_func(x=2):
   """Do some math."""
   return x + x


result = addition_func(4)