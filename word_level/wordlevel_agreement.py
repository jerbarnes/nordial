import nltk
from nltk.metrics import masi_distance
from nltk.metrics import jaccard_distance

"""
Data should have the following format: for each token, include
a tuple of (annotator_id, token_id, frozenset of labels). Each token needs
to have an individual token_id. The order of the labels doesn't matter, as the
agreement function operates on sets.
"""

task_data = [('coder1', 'Item0', frozenset(['l1', 'l2'])),
             ('coder2', 'Item0', frozenset(['l2', 'l1'])),
             ("coder3", "Item0", frozenset(['l1', 'l2'])),
             # -----------------------------------------------------
             ("coder1", "Item1", frozenset(['l1'])),
             ("coder2", "Item1", frozenset(['l1'])),
             ("coder3", "Item1", frozenset(['l1'])),
             # -----------------------------------------------------
             ("coder1", "Item2", frozenset(['l1', 'l2'])),
             ("coder2", "Item2", frozenset(['l2', 'l1'])),
             ("coder3", "Item2", frozenset(['l1', 'l2'])),
             # -----------------------------------------------------
             ("coder1", "Item3", frozenset(['l3', 'l1'])),
             ("coder2", "Item3", frozenset(['l2', 'l3'])),
             ("coder3", "Item3", frozenset(['l3', 'l1']))
             ]

task = nltk.agreement.AnnotationTask(data=task_data, distance=masi_distance)
print(task.alpha())
