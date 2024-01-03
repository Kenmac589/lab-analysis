
>>> How to I append a new key and related list to a dictionnary in python

To append a new key and related list to a dictionary in Python, you can use the `update()` method.
Here's an example:
```
d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
new_key = 'c'
new_list = [7, 8, 9]
d.update({new_key: new_list})
print(d) # Output: {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
```
Alternatively, you can use the `setdefault()` method to create a new key and associated list if it
doesn't already exist. Here's an example:
```
d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
new_key = 'c'
new_list = [7, 8, 9]
d.setdefault(new_key, new_list)
print(d) # Output: {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
```
Both of these methods will add the new key and list to the dictionary, creating a new entry if it
doesn't already exist.

>>> do all keys in a dictionnary have to be unique

No, all keys in a dictionary do not have to be unique. In Python, a key can appear more than once
in a dictionary, and each occurrence of the key will point to its own separate value. This is known
as a "duplicate key" or "collision."

For example:
```
d = {'a': 1, 'b': 2, 'c': 3, 'a': 4}
print(d) # Output: {'a': 1, 'b': 2, 'c': 3, 'a': 4}
```
In this example, the key `'a'` appears twice in the dictionary. This is legal in Python, and each
occurrence of the key will point to its own separate value. However, it's important to note that
when you try to access a value using a duplicate key, you may not get the result you expect. For
example:
```
print(d['a']) # Output: 1
print(d['a']) # Output: 4
```
In this case, `d['a']` returns `1`, but if you try to access it again using the same key, it will
return `4`. This can make it difficult to keep track of which value is associated with each key.

It's generally a good idea to avoid using duplicate keys in your dictionaries, as they can make
your code more complex and harder to read. Instead, use unique keys for each value you want to
store in the dictionary.

>>> is there a way to control the order in which dupicate keys are printed for example using a key that o
... ccurs multiple time as a tag more for filtering only certain desired lists in a dictionary

Yes, there are ways to control the order in which duplicate keys are printed when iterating over a
dictionary. One way is to use the `items()` method of the dictionary, which returns an iterator
that generates key-value pairs in the order they were inserted into the dictionary:
```
d = {'a': 1, 'b': 2, 'c': 3, 'a': 4}
for key, value in d.items():
    print(f"{key}: {value}")
# Output: a: 1, b: 2, c: 3, a: 4
```
In this example, the order of the keys is determined by the order they were inserted into the
dictionary. If you want to control the order of the keys, you can use the `sorted()` function to
sort the keys before iterating over them:
```
d = {'a': 1, 'b': 2, 'c': 3, 'a': 4}
for key in sorted(d.keys()):
    print(f"{key}: {d[key]}")
# Output: a: 1, a: 4, b: 2, c: 3
```
In this example, the keys are sorted alphabetically before being printed. You can use any function
that takes a sequence of strings as input to sort the keys in a specific order. For example, you
could use `sorted(d.keys(), key=lambda x: x.lower())` to sort the keys by lowercase value.

Another way to control the order of duplicate keys is to use the `dict.fromkeys()` method, which
creates a new dictionary with specified keys and values, in the order they are given. Here's an
example:
```
d = {'a': 1, 'b': 2, 'c': 3, 'a': 4}
new_dict = dict.fromkeys(sorted(d.keys()), None)
print(new_dict) # Output: {'a': None, 'a': None, 'b': None, 'c': None}
```
In this example, the keys are sorted alphabetically before being used to create a new dictionary
with `None` values for each key. This allows you to control the order of the keys and still
preserve the original values associated with each key.

