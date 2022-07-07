class data_dictionary(object):
    """
    class that temporarily stores and retrieves data when needed
    """
    def __init__(self):
        self.data = {}  # just saying a data or anything is equal to a dictionary {} or a []

    def __setitem__(self, key, data):  # then setting the key section of the dictionary to 'data'
        self.data[key] = data

    def __getitem__(self, key): # how to get the part that was set as key
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def clear(self):
        self.data.clear()




