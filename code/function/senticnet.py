# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:02:23 2017

@author: wqmike123

"""

import importlib

class Senticnet(object):
    """
    Simple API to use Senticnet 4.
    """
    def __init__(self, babel_dir = None,language="en"):
        if not babel_dir:
            babel_dir = r'C:\Users\wqmike123\Documents\thesis\dictionary\\'
        loader = importlib.machinery.SourceFileLoader('data_'+language,babel_dir+'babel\data_'+language+'.py')
        data_module = loader.load_module()
        self.data = data_module.senticnet

    # public methods

    def concept(self, concept):
        """
        Return all the information about a concept: semantics,
        sentics and polarity.
        """
        result = {}

        result["polarity_value"] = self.polarity_value(concept)
        result["polarity_intense"] = self.polarity_intense(concept)
        result["moodtags"] = self.moodtags(concept)
        result["sentics"] = self.sentics(concept)
        result["semantics"] = self.semantics(concept)

        return result

    def semantics(self, concept):
        """
        Return the semantics associated with a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[8:]

    def sentics(self, concept):
        """
        Return sentics of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        sentics = {"pleasantness": concept_info[0],
                   "attention": concept_info[1],
                   "sensitivity": concept_info[2],
                   "aptitude": concept_info[3]}

        return sentics

    def polarity_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[6]

    def polarity_intense(self, concept):
        """
        Return the polarity intense of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[7]

    def moodtags(self, concept):
        """
        Return the moodtags of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[4:6]
    
if __name__=='__main__':
    
    sn = Senticnet()
    print("polarity value:", sn.polarity_value("love"))
    print("polarity intense:", sn.polarity_intense("love"))
    print("moodtags:", ", ".join(sn.moodtags("love")))
    print("semantics:", ", ".join(sn.semantics("love")))
    print("\n".join([key + ": " + str(value) for key, value in sn.sentics("love").items()]))