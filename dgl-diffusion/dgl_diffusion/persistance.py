import hashlib
import os
import sys
from itertools import chain

class PManager:
    def __init__(self, save_dir, force_overwrite = False):
        """
        Attributes
        ----------
        h : hashlib
          hash function

        hashed_value : list of tuples (str, str)

        tree_generated : bool
          if true, the directory structure has already
          been initialized
          
        Parameters
        ----------
        save_dir : str
          path of the root directory

        force_overwrite : bool, default False
          if True the folder will update without asking
          the user for permission
        """
        self.save_dir = save_dir
        self.h = hashlib.sha256()
        self.hashed_values = []
        self.tree_generated = False
        self.force_overwrite = force_overwrite


    def hash(self, *value):
        """Add an element to
        the hash computation

        Parameters
        ----------
        value : any, multiple
          any object that can be converted as string
        
        """
        if not self.save_dir:
            return

        if self.tree_generated:
            raise ValueError
        
        for v in value:
            if isinstance(v, tuple):
                v_name, v_str = map(str, v)
                self.hashed_values.append((v_name, v_str))
            else:
                v_str = str(v)
            self.h.update(b"{v_str}")

    def hex(self):
        return self.h.hexdigest()

    def generate_tree(self):
        """Initialize the tree
        structure where to save the
        """
        if not self.save_dir:
            return
            
        dir_name = self.h.hexdigest()
        # update the save directory
        self.save_dir = os.path.join(self.save_dir, dir_name)
        # create the directory
        try: 
            os.makedirs(self.save_dir)
        except FileExistsError:
            while not self.force_overwrite:
                print(f"{self.save_dir} already exists. Override? [y/n]")
                decision = input()
                decision = decision.lower()
                if decision not in ("y", "n"):
                    print(f"{decision} What?????")
                    continue
                elif decision == "y":
                    os.makedirs(self.save_dir, exist_ok=True)
                    break
                else:
                    print("\033[91mRefuse to override. No Persistence\033[0m'")
                    self.save_dir = None
                    break

        # freeze the hash
        self.tree_generated = True

    def persist(self, *name_callable_mode):
        if not self.tree_generated:
            self.generate_tree()

        # no saving to disk
        if not self.save_dir:
            return

        for v in name_callable_mode:
            if not isinstance(v, tuple):
                raise RuntimeError
            file_name, fn, mode = v
            with open(os.path.join(self.save_dir, file_name), mode) as f:
                fn(f)


    def close(self, *additional_info):
        if not self.tree_generated:
            self.generate_tree()

        if not self.save_dir:
            return
        if not self.hashed_values:
            return
        
        with open(os.path.join(self.save_dir, "readme.txt"), 'w') as f:
            for pname, pvalue in chain(self.hashed_values, additional_info):
                f.write(f"{pname}:{pvalue}\n")

