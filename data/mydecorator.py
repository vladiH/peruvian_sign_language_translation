
# coding: utf-8

# In[1]:

import functools

#https://danijar.com/structuring-your-tensorflow-models/
def scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

