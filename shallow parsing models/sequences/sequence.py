import sys
import pdb

class Sequence(object):
    def __init__(self, sequence_list, x, y, pos, nr,file_id,br_pos=[]):
        self.x = x
        self.y = y
        self.pos = pos
        self.nr = nr
        self.file_id = file_id
        self.br_positions = br_pos
        self.sequence_list = sequence_list


    def size(self):
        '''Returns the size of the sequence.'''
        return len(self.x)

    def __len__(self):
        return len(self.x)

    def copy_sequence(self):
        '''Performs a deep copy of the sequence'''
        s = Sequence(self.sequence_list, self.x[:], self.y[:], self.pos[:],self.nr,self.file_id,self.br_positions)
        return s


    def update_from_sequence(self,new_y):
        '''Returns a new sequence equal to the previous but with y set to newy'''
        s = Sequence(self.sequence_list, self.x, new_y, self.pos, self.nr,self.file_id,self.br_positions)
        return s

    def __str__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            rep += "%i: %s/%s\n" % (i, self.sequence_list.x_dict.get_label_name(xi),
                                self.sequence_list.y_dict.get_label_name(yi))
        return rep

    def __repr__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            rep += "%i : %s/%s\n"%(i,self.sequence_list.x_dict.get_label_name(xi),
                             self.sequence_list.y_dict.get_label_name(yi))
        return rep
