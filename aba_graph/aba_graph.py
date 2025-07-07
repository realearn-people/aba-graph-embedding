from py_arg.aba_classes.rule import Rule
from py_arg.aba_classes.aba_framework import ABAF
from py_arg.aba_classes.semantics.get_preferred_extensions import get_preferred_extensions
from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat

import json
import matplotlib.pyplot as plt
import networkx as nx
import random
import sys 
import os


class ABA_Graph:
    def __init__(self):
        # Set of logical symbols
        self.language = set()
        # Set of inference rules
        self.rules = set()
        # Set of assumptions
        self.assumptions = set()
        # Contrary functions
        self.contraries = {}
        

    def load_json(self, json_data):
        """
        Load a JSON document containing the framework elements.
        """

        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        self.language = set(data.get('language', []))
        self.assumptions = set(data.get('assumptions', []))
        self.contraries = data.get('contraries', {})

        self.rules = {
            Rule(f'Rule_{i+1}', set(rule['body']), rule['head'])
            for i, rule in enumerate(data.get('rules', []))
        }

        return True

    
    def create_aba_framework(self):
        """
        Building an ABA framework
        """
        return ABAF(self.assumptions, self.rules, self.language, self.contraries)
    
    
    def aba_to_aaf(self, aba_framework):
        """
        Converts an ABA framework to Abstract Argumentation Framework
        """
        # ABA argument creation: generates all the arguments of the ABA framework
        # Support: set of assumptions used and conclusion: deduced formula
        aba_arguments = self.generate_arguments_from_framework(aba_framework)
        
        # Creation of abstract arguments
        abstract_arguments = []
        for support, conclusion in aba_arguments:
            arg_name = f"{', '.join(support)} ⊢ {conclusion}" if support else conclusion
            abstract_arguments.append(Argument(arg_name))
        
        # Creation of attacks
        defeats = []
        # Comparaison of each pair of ABA arguments
        #i is the opponent and j is the proponent
        for i, (support_i, conclusion_i) in enumerate(aba_arguments):
            for j, (support_j, conclusion_j) in enumerate(aba_arguments):
                # Checks if conclusion_i is the opposite of an assumption in support_j
                for assumption in support_j:
                    # If the argument i concludes ¬a 
                    # And that a is an assumption used in the support of argument j
                    # Then i attacks j (undercut)
                    if assumption in aba_framework.contraries and aba_framework.contraries[assumption] == conclusion_i:
                        # We therefore add an attack from argument i to j
                        defeats.append(Defeat(abstract_arguments[i], abstract_arguments[j]))
        
        return AbstractArgumentationFramework('ABA_AF', abstract_arguments, defeats)
    

    def generate_arguments_from_framework(self, aba_framework):
        """
        Generates all possible arguments from the ABA framework.
        Each argument is represented as a tuple: (support, conclusion).

        Support: set of assumptions {a_1, a_2, ..., a_n}
        Conclusion: deduction using rules and assumptions
        """

        arguments = []
        
        for assumption in aba_framework.assumptions:
            arguments.append(({assumption}, assumption))
        
        changed = True
        while changed:
            changed = False
            new_arguments = []
            
            for rule in aba_framework.rules:
                # Checks whether all the premises can be supported
                supported_premises = []
                for premise in rule.body:
                    found = False
                    for support, conclusion in arguments:
                        if conclusion == premise:
                            supported_premises.append((support, conclusion))
                            found = True
                            break
                    if not found:
                        break
                
                if len(supported_premises) == len(rule.body):
                    # All premises are supported, we can create a new argument
                    new_support = set()
                    for s, _ in supported_premises:
                        new_support.update(s)
                    
                    # Verifying if it's a new argument
                    new_arg = (new_support, rule.head)
                    if new_arg not in arguments:
                        new_arguments.append(new_arg)
                        changed = True
            arguments.extend(new_arguments)

        return arguments
    

    def visualize(self, aba_framework, show_extensions=True):
        af = self.aba_to_aaf(aba_framework)
        
        G = nx.DiGraph()
        
        # Adding arguments as nodes
        for arg in af.arguments:
            G.add_node(arg.name)
        
        # Adding attacks as edges
        for defeat in af.defeats:
            G.add_edge(defeat.from_argument.name, defeat.to_argument.name)
        
        # Drawing the graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        
        # Drawing the nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', node_shape='o')
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Drawing the edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, arrowstyle='-|>')
        plt.title("ABA Graph Visualization")
        plt.axis('off')
        
        # Showing the extensions
        if show_extensions:
            extensions = get_preferred_extensions(aba_framework)
            ext_text = "\n".join([f"Extension {i+1}: {ext}" 
                                for i, ext in enumerate(extensions)])
            plt.figtext(0.5, 0.01, f"Preferred Extensions:\n{ext_text}", 
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        plt.show()


if __name__ == "__main__":
    path = "./json/ex3_1.json"

    with open(path, 'r') as f:
        aba = json.load(f)

    build = ABA_Graph()
    if build.load_json(aba):
        aba_framework = build.create_aba_framework()
        build.visualize(aba_framework)        

