"""
AI-based Medical Diagnostic Assistant
======================================

This module implements an AI-based diagnostic assistant for a local clinic.
The system receives a set of symptoms from a patient and uses a knowledge base
of rules to infer the most likely diseases using both forward and backward
reasoning techniques.

Features:
- Knowledge base representation with rules and facts
- Forward reasoning implementation (data-driven)
- Backward reasoning implementation (goal-driven)
- Visualization of inference chains and reasoning process
- Performance comparison between reasoning approaches
- Example medical knowledge base and symptom sets

Author: Kodu AI
Date: April 2025
"""

import time
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import numpy as np
from matplotlib.colors import ListedColormap


class Rule:
    """
    Represents a rule in the knowledge base.
    
    A rule consists of a set of antecedents (conditions) and a set of consequents
    (conclusions). If all antecedents are true, then all consequents become true.
    
    Attributes:
        antecedents (set): Set of antecedent conditions
        consequents (set): Set of consequent conclusions
        name (str): Name/identifier of the rule
        confidence (float): Confidence value between 0.0 and 1.0
    """
    
    def __init__(self, antecedents, consequents, name=None, confidence=1.0):
        """
        Initialize a rule.
        
        Args:
            antecedents (list or set): List/set of antecedent conditions
            consequents (list or set): List/set of consequent conclusions
            name (str, optional): Name/identifier for the rule
            confidence (float, optional): Confidence value between 0.0 and 1.0
        """
        self.antecedents = set(antecedents)
        self.consequents = set(consequents)
        self.name = name or f"R{random.randint(1000, 9999)}"
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
    def __str__(self):
        """String representation of the rule."""
        ante_str = " AND ".join(self.antecedents)
        cons_str = " AND ".join(self.consequents)
        return f"{self.name}: IF {ante_str} THEN {cons_str} [conf={self.confidence:.2f}]"
    
    def __repr__(self):
        """Representation of the rule."""
        return self.__str__()


class KnowledgeBase:
    """
    Represents a knowledge base containing facts and rules.
    
    Attributes:
        facts (set): Set of known facts
        rules (list): List of rules
        fact_explanations (dict): Maps facts to explanations of how they were derived
    """
    
    def __init__(self):
        """Initialize an empty knowledge base."""
        self.facts = set()
        self.rules = []
        self.fact_explanations = {}
    
    def add_fact(self, fact, explanation=None):
        """
        Add a fact to the knowledge base.
        
        Args:
            fact (str): The fact to add
            explanation (str, optional): Explanation for the fact
        """
        self.facts.add(fact)
        if explanation:
            self.fact_explanations[fact] = explanation
    
    def add_rule(self, rule):
        """
        Add a rule to the knowledge base.
        
        Args:
            rule (Rule): The rule to add
        """
        self.rules.append(rule)
    
    def is_fact(self, fact):
        """
        Check if something is a known fact.
        
        Args:
            fact (str): The fact to check
            
        Returns:
            bool: True if it is a known fact, False otherwise
        """
        return fact in self.facts
    
    def explain_fact(self, fact):
        """
        Get explanation for how a fact was derived.
        
        Args:
            fact (str): The fact to explain
            
        Returns:
            str: Explanation for the fact, or a default message
        """
        if fact not in self.facts:
            return f"'{fact}' is not a known fact."
        
        return self.fact_explanations.get(fact, f"'{fact}' is a base fact (given or asserted).")
    
    def get_facts_for_visualization(self):
        """
        Get facts with metadata for visualization.
        
        Returns:
            list: List of facts with metadata
        """
        return [
            {
                'name': fact,
                'explanation': self.explain_fact(fact),
                'is_derived': fact in self.fact_explanations
            }
            for fact in self.facts
        ]


class ForwardReasoning:
    """
    Implements forward reasoning (data-driven) on a knowledge base.
    
    Forward reasoning starts with known facts and applies rules to derive new facts
    until no more facts can be derived or a goal is reached.
    
    Attributes:
        kb (KnowledgeBase): The knowledge base to reason with
        inference_trace (list): Trace of the inference process (for visualization)
    """
    
    def __init__(self, kb):
        """
        Initialize the forward reasoner.
        
        Args:
            kb (KnowledgeBase): The knowledge base to reason with
        """
        self.kb = kb
        self.inference_trace = []
    
    def infer(self, max_iterations=100):
        """
        Perform forward reasoning on the knowledge base.
        
        Args:
            max_iterations (int, optional): Maximum number of reasoning iterations
            
        Returns:
            set: Set of newly derived facts
        """
        # Start time
        start_time = time.time()
        
        # Initialize
        new_facts = set()
        iterations = 0
        
        # Clear previous inference trace
        self.inference_trace = []
        
        # Save initial state
        self.inference_trace.append({
            'iteration': 0,
            'facts': self.kb.facts.copy(),
            'new_facts': set(),
            'rule_applied': None
        })
        
        # Main inference loop
        while iterations < max_iterations:
            iterations += 1
            current_facts = self.kb.facts.copy()
            derived_facts = set()
            applied_rule = None
            
            # Try to apply each rule
            for rule in self.kb.rules:
                # Check if all antecedents are satisfied
                if rule.antecedents.issubset(current_facts):
                    # Add consequents to derived facts
                    for consequent in rule.consequents:
                        if consequent not in current_facts:
                            derived_facts.add(consequent)
                            self.kb.add_fact(
                                consequent,
                                f"Derived from rule: {rule.name} "
                                f"with confidence {rule.confidence:.2f}"
                            )
                            new_facts.add(consequent)
                    
                    # If we derived new facts, record the rule
                    if derived_facts:
                        applied_rule = rule
            
            # Save state for this iteration
            self.inference_trace.append({
                'iteration': iterations,
                'facts': self.kb.facts.copy(),
                'new_facts': derived_facts,
                'rule_applied': applied_rule
            })
            
            # If no new facts were derived, we're done
            if not derived_facts:
                break
        
        # End time
        self.execution_time = time.time() - start_time
        
        return new_facts
    
    def explain_inference(self):
        """
        Generate a textual explanation of the inference process.
        
        Returns:
            str: Explanation of the inference process
        """
        if not self.inference_trace:
            return "No inference has been performed yet."
        
        explanation = ["Forward Reasoning Inference Process:"]
        
        # Initial facts
        initial_facts = self.inference_trace[0]['facts']
        explanation.append(f"Initial facts: {', '.join(initial_facts)}")
        
        # Inference steps
        for i in range(1, len(self.inference_trace)):
            step = self.inference_trace[i]
            
            if step['rule_applied'] and step['new_facts']:
                explanation.append(
                    f"Iteration {step['iteration']}: Applied rule {step['rule_applied'].name} "
                    f"and derived {', '.join(step['new_facts'])}"
                )
            else:
                explanation.append(f"Iteration {step['iteration']}: No new facts derived.")
        
        # Final facts
        final_facts = self.inference_trace[-1]['facts']
        explanation.append(f"Final facts: {', '.join(final_facts)}")
        
        return "\n".join(explanation)
    
    def visualize_inference(self):
        """Visualize the forward reasoning inference process."""
        if not self.inference_trace:
            print("No inference to visualize. Run infer() first.")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add initial facts (as initial nodes)
        initial_facts = self.inference_trace[0]['facts']
        for fact in initial_facts:
            G.add_node(fact, type='initial_fact')
        
        # Add derived facts and rules as the inference progresses
        for i in range(1, len(self.inference_trace)):
            step = self.inference_trace[i]
            rule = step['rule_applied']
            
            if rule and step['new_facts']:
                # Add rule node
                rule_node = f"{rule.name}"
                G.add_node(rule_node, type='rule')
                
                # Connect antecedents to rule
                for antecedent in rule.antecedents:
                    G.add_edge(antecedent, rule_node)
                
                # Connect rule to consequents
                for consequent in rule.consequents:
                    if consequent in step['new_facts']:
                        G.add_node(consequent, type='derived_fact')
                        G.add_edge(rule_node, consequent)
        
        # Define node colors
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('type') == 'initial_fact':
                node_colors.append('lightgreen')
            elif G.nodes[node].get('type') == 'derived_fact':
                node_colors.append('lightblue')
            else:  # rule
                node_colors.append('orange')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                       markersize=15, label='Initial Facts'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                       markersize=15, label='Applied Rules'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                       markersize=15, label='Derived Facts')
        ]
        plt.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, 1.1), ncol=3)
        
        plt.title('Forward Reasoning Inference Chain')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class BackwardReasoning:
    """
    Implements backward reasoning (goal-driven) on a knowledge base.
    
    Backward reasoning starts with a goal and works backwards to see if it can be
    supported by the known facts, potentially discovering subgoals along the way.
    
    Attributes:
        kb (KnowledgeBase): The knowledge base to reason with
        inference_trace (list): Trace of the inference process (for visualization)
        proof_trace (dict): Trace of the proof for each goal
    """
    
    def __init__(self, kb):
        """
        Initialize the backward reasoner.
        
        Args:
            kb (KnowledgeBase): The knowledge base to reason with
        """
        self.kb = kb
        self.inference_trace = []
        self.proof_trace = {}
    
    def prove(self, goal, visited=None, depth=0, max_depth=10):
        """
        Attempt to prove a goal using backward reasoning.
        
        Args:
            goal (str): The goal to prove
            visited (set, optional): Set of already visited goals
            depth (int, optional): Current recursion depth
            max_depth (int, optional): Maximum recursion depth
            
        Returns:
            tuple: (bool, dict) Whether the goal was proven and explanation
        """
        # Avoid infinite recursion
        if visited is None:
            visited = set()
        
        if goal in visited or depth > max_depth:
            return False, {"status": "recursion_limit", "goal": goal}
        
        visited.add(goal)
        
        # Record step in inference trace
        self.inference_trace.append({
            'step': len(self.inference_trace) + 1,
            'goal': goal,
            'depth': depth,
            'status': 'unknown'
        })
        
        # Check if goal is already a known fact
        if self.kb.is_fact(goal):
            # Update step status
            self.inference_trace[-1]['status'] = 'fact'
            
            # Return proof
            return True, {
                "status": "fact", 
                "goal": goal, 
                "explanation": self.kb.explain_fact(goal)
            }
        
        # Find rules with the goal in their consequents
        relevant_rules = [
            rule for rule in self.kb.rules
            if goal in rule.consequents
        ]
        
        # If no relevant rules, goal can't be proven
        if not relevant_rules:
            # Update step status
            self.inference_trace[-1]['status'] = 'no_rules'
            
            return False, {"status": "no_rules", "goal": goal}
        
        # Try each rule
        for rule in relevant_rules:
            # Record rule in inference trace
            self.inference_trace[-1]['rule'] = rule.name
            
            # Try to prove all antecedents
            all_antecedents_proven = True
            antecedent_proofs = {}
            
            for antecedent in rule.antecedents:
                proven, proof = self.prove(antecedent, visited.copy(), depth + 1, max_depth)
                
                if not proven:
                    all_antecedents_proven = False
                    break
                
                antecedent_proofs[antecedent] = proof
            
            # If all antecedents proven, goal is proven
            if all_antecedents_proven:
                # Update step status
                self.inference_trace[-1]['status'] = 'proven'
                
                # Add the goal as a fact if it wasn't already
                if not self.kb.is_fact(goal):
                    self.kb.add_fact(
                        goal,
                        f"Derived by backward reasoning using rule: {rule.name} "
                        f"with confidence {rule.confidence:.2f}"
                    )
                
                # Return proof
                proof = {
                    "status": "derived", 
                    "goal": goal, 
                    "rule": rule.name, 
                    "antecedents": antecedent_proofs,
                    "confidence": rule.confidence
                }
                
                # Save proof for later
                self.proof_trace[goal] = proof
                
                return True, proof
        
        # If we get here, no rule could be used to prove the goal
        # Update step status
        self.inference_trace[-1]['status'] = 'unprovable'
        
        return False, {"status": "unprovable", "goal": goal}
    
    def infer(self, goals):
        """
        Attempt to prove multiple goals using backward reasoning.
        
        Args:
            goals (list): List of goals to prove
            
        Returns:
            dict: Dictionary mapping goals to proof results
        """
        # Start time
        start_time = time.time()
        
        # Clear previous inference trace
        self.inference_trace = []
        self.proof_trace = {}
        
        # Try to prove each goal
        results = {}
        for goal in goals:
            proven, proof = self.prove(goal)
            results[goal] = {"proven": proven, "proof": proof}
        
        # End time
        self.execution_time = time.time() - start_time
        
        return results
    
    def explain_proof(self, goal):
        """
        Generate a textual explanation of the proof for a goal.
        
        Args:
            goal (str): The goal to explain
            
        Returns:
            str: Explanation of the proof
        """
        if goal not in self.proof_trace:
            return f"No proof found for '{goal}'."
        
        proof = self.proof_trace[goal]
        
        # Format the proof recursively
        def format_proof(proof, indent=0):
            indentation = "  " * indent
            result = []
            
            if proof["status"] == "fact":
                result.append(f"{indentation}FACT: {proof['goal']} - {proof['explanation']}")
            
            elif proof["status"] == "derived":
                result.append(f"{indentation}DERIVED: {proof['goal']} using rule {proof['rule']}")
                result.append(f"{indentation}WITH ANTECEDENTS:")
                for antecedent, antecedent_proof in proof["antecedents"].items():
                    result.append(f"{indentation}  • {antecedent}:")
                    result.extend(format_proof(antecedent_proof, indent + 2))
            
            return result
        
        explanation_lines = format_proof(proof)
        return "\n".join(explanation_lines)
    
    def explain_inference(self, results):
        """
        Generate a textual explanation of the inference process for multiple goals.
        
        Args:
            results (dict): Results from infer()
            
        Returns:
            str: Explanation of the inference process
        """
        explanation = ["Backward Reasoning Inference Process:"]
        
        for goal, result in results.items():
            if result["proven"]:
                explanation.append(f"Goal '{goal}' was PROVEN.")
                explanation.append(self.explain_proof(goal))
            else:
                explanation.append(f"Goal '{goal}' could NOT be proven.")
                
                proof = result["proof"]
                if proof["status"] == "no_rules":
                    explanation.append(f"  No rules were found with '{goal}' as a consequent.")
                elif proof["status"] == "recursion_limit":
                    explanation.append(f"  Recursion limit reached while trying to prove '{goal}'.")
                elif proof["status"] == "unprovable":
                    explanation.append(f"  All relevant rules for '{goal}' had unprovable antecedents.")
            
            explanation.append("")
        
        return "\n".join(explanation)
    
    def visualize_proof(self, goal):
        """
        Visualize the proof tree for a goal.
        
        Args:
            goal (str): The goal to visualize
        """
        if goal not in self.proof_trace:
            print(f"No proof found for '{goal}'.")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges recursively
        def add_proof_to_graph(proof, parent=None):
            node_id = proof["goal"]
            
            # Add the node
            if proof["status"] == "fact":
                G.add_node(node_id, type='fact')
            else:  # derived
                G.add_node(node_id, type='derived', rule=proof["rule"])
            
            # Connect to parent if exists
            if parent:
                G.add_edge(node_id, parent)
            
            # Add antecedents if derived
            if proof["status"] == "derived":
                for antecedent, antecedent_proof in proof["antecedents"].items():
                    add_proof_to_graph(antecedent_proof, node_id)
        
        # Build the graph
        add_proof_to_graph(self.proof_trace[goal])
        
        # Define node colors
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('type') == 'fact':
                node_colors.append('lightgreen')
            else:  # derived
                node_colors.append('lightblue')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add rule labels to the edges
        edge_labels = {}
        for node in G.nodes():
            if G.nodes[node].get('type') == 'derived':
                for neighbor in G.neighbors(node):
                    edge_labels[(node, neighbor)] = G.nodes[node].get('rule', '')
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                       markersize=15, label='Facts'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                       markersize=15, label='Derived Goals')
        ]
        plt.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, 1.1), ncol=2)
        
        plt.title(f'Backward Reasoning Proof Tree for: {goal}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_inference(self):
        """Visualize the backward reasoning inference process."""
        if not self.inference_trace:
            print("No inference to visualize. Run infer() first.")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Group steps by goal
        goal_steps = {}
        for step in self.inference_trace:
            goal = step['goal']
            if goal not in goal_steps:
                goal_steps[goal] = []
            goal_steps[goal].append(step)
        
        # Add nodes for each goal
        for goal, steps in goal_steps.items():
            # Determine node type based on final status
            final_step = steps[-1]
            status = final_step['status']
            
            node_type = 'unknown'
            if status == 'fact':
                node_type = 'fact'
            elif status == 'proven':
                node_type = 'proven'
            elif status == 'unprovable' or status == 'no_rules':
                node_type = 'unprovable'
            
            G.add_node(goal, type=node_type, depth=final_step['depth'])
        
        # Add edges based on rule relationships
        for step in self.inference_trace:
            if 'rule' in step:
                # Find the rule
                rule_name = step['rule']
                rule = next((r for r in self.kb.rules if r.name == rule_name), None)
                
                if rule:
                    # Connect antecedents to goal
                    for antecedent in rule.antecedents:
                        if antecedent in G.nodes and step['goal'] in G.nodes:
                            G.add_edge(antecedent, step['goal'], rule=rule_name)
        
        # Define node colors
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type')
            if node_type == 'fact':
                node_colors.append('lightgreen')
            elif node_type == 'proven':
                node_colors.append('lightblue')
            else:  # unprovable or unknown
                node_colors.append('lightcoral')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create layout based on depth
        pos = {}
        for node in G.nodes():
            depth = G.nodes[node].get('depth', 0)
            pos[node] = (depth, random.uniform(0, 1))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add rule labels to the edges
        edge_labels = {(u, v): d.get('rule', '') for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                       markersize=15, label='Known Facts'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                       markersize=15, label='Proven Goals'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                       markersize=15, label='Unprovable Goals')
        ]
        plt.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, 1.1), ncol=3)
        
        plt.title('Backward Reasoning Inference Process')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class DiagnosticAssistant:
    """
    AI-based diagnostic assistant that combines forward and backward reasoning.
    
    This assistant uses a medical knowledge base to infer possible diagnoses from
    patient symptoms using both forward and backward reasoning techniques.
    
    Attributes:
        kb (KnowledgeBase): The medical knowledge base
        forward (ForwardReasoning): Forward reasoning engine
        backward (BackwardReasoning): Backward reasoning engine
    """
    
    def __init__(self):
        """Initialize the diagnostic assistant."""
        self.kb = KnowledgeBase()
        self.forward = ForwardReasoning(self.kb)
        self.backward = BackwardReasoning(self.kb)
        
        # Add medical knowledge base (rules, diseases, symptoms)
        self._build_medical_knowledge_base()
    
    def _build_medical_knowledge_base(self):
        """Build a sample medical knowledge base with rules and facts."""
        # Common cold rule
        self.kb.add_rule(Rule(
            ["runny_nose", "sneezing", "sore_throat"],
            ["common_cold"],
            name="R1", confidence=0.9
        ))
        
        # Flu rules
        self.kb.add_rule(Rule(
            ["fever", "muscle_ache", "fatigue"],
            ["influenza"],
            name="R2", confidence=0.8
        ))
        
        self.kb.add_rule(Rule(
            ["common_cold", "fever", "fatigue"],
            ["influenza"],
            name="R3", confidence=0.7
        ))
        
        # Allergic rhinitis rule
        self.kb.add_rule(Rule(
            ["runny_nose", "sneezing", "itchy_eyes"],
            ["allergic_rhinitis"],
            name="R4", confidence=0.85
        ))
        
        # Bacterial infection rules
        self.kb.add_rule(Rule(
            ["fever", "green_phlegm", "persistent_cough"],
            ["bacterial_infection"],
            name="R5", confidence=0.75
        ))
        
        self.kb.add_rule(Rule(
            ["sore_throat", "white_spots_throat", "swollen_glands"],
            ["strep_throat"],
            name="R6", confidence=0.9
        ))
        
        # Sinusitis rules
        self.kb.add_rule(Rule(
            ["facial_pain", "headache", "green_mucus"],
            ["sinusitis"],
            name="R7", confidence=0.8
        ))
        
        self.kb.add_rule(Rule(
            ["common_cold", "facial_pain", "headache"],
            ["sinusitis"],
            name="R8", confidence=0.7
        ))
        
        # Bronchitis rules
        self.kb.add_rule(Rule(
            ["persistent_cough", "chest_discomfort", "phlegm"],
            ["bronchitis"],
            name="R9", confidence=0.85
        ))
        
        self.kb.add_rule(Rule(
            ["influenza", "persistent_cough", "phlegm"],
            ["bronchitis"],
            name="R10", confidence=0.8
        ))
        
        # More complex interdependent rules
        self.kb.add_rule(Rule(
            ["bronchitis", "fever", "shortness_of_breath"],
            ["pneumonia"],
            name="R11", confidence=0.9
        ))
        
        self.kb.add_rule(Rule(
            ["bacterial_infection", "persistent_cough", "shortness_of_breath"],
            ["pneumonia"],
            name="R12", confidence=0.85
        ))
        
        # Migraine rules
        self.kb.add_rule(Rule(
            ["headache", "nausea", "light_sensitivity"],
            ["migraine"],
            name="R13", confidence=0.9
        ))
        
        self.kb.add_rule(Rule(
            ["headache", "aura", "one_sided_pain"],
            ["migraine"],
            name="R14", confidence=0.95
        ))
        
        # Rules for emergency conditions
        self.kb.add_rule(Rule(
            ["sudden_severe_headache", "neck_stiffness", "confusion"],
            ["meningitis"],
            name="R15", confidence=0.95
        ))
        
        self.kb.add_rule(Rule(
            ["chest_pain", "shortness_of_breath", "sweating"],
            ["heart_attack"],
            name="R16", confidence=0.95
        ))
    
    def set_symptoms(self, symptoms):
        """
        Set patient symptoms as facts in the knowledge base.
        
        Args:
            symptoms (list): List of symptom strings
            
        Returns:
            list: List of added symptoms
        """
        # Clear existing facts
        self.kb.facts = set()
        self.kb.fact_explanations = {}
        
        # Add each symptom as a fact
        for symptom in symptoms:
            self.kb.add_fact(symptom, "Patient-reported symptom")
        
        return symptoms
    
    def diagnose_forward(self):
        """
        Perform diagnosis using forward reasoning.
        
        Returns:
            tuple: (derived_facts, execution_time)
        """
        # Perform forward reasoning
        derived_facts = self.forward.infer()
        
        # Extract diagnoses (non-symptoms)
        diagnoses = [
            fact for fact in derived_facts
            if not fact.startswith("test_") and fact not in self.kb.fact_explanations
        ]
        
        return {
            'diagnoses': diagnoses,
            'execution_time': self.forward.execution_time,
            'all_derived': derived_facts,
            'all_facts': self.kb.facts
        }
    
    def diagnose_backward(self, possible_diseases=None):
        """
        Perform diagnosis using backward reasoning.
        
        Args:
            possible_diseases (list, optional): List of diseases to check,
                if None, checks all diseases in the knowledge base
                
        Returns:
            tuple: (results, execution_time)
        """
        # If no diseases specified, extract all disease consequents from rules
        if possible_diseases is None:
            possible_diseases = set()
            for rule in self.kb.rules:
                for consequent in rule.consequents:
                    if not consequent.endswith("_symptom"):
                        possible_diseases.add(consequent)
        
        # Perform backward reasoning
        results = self.backward.infer(possible_diseases)
        
        # Extract confirmed diagnoses
        confirmed_diagnoses = [
            disease for disease, result in results.items()
            if result["proven"]
        ]
        
        return {
            'diagnoses': confirmed_diagnoses,
            'execution_time': self.backward.execution_time,
            'all_results': results
        }
    
    def compare_reasoning_approaches(self, symptoms, possible_diseases=None):
        """
        Compare forward and backward reasoning for diagnosis.
        
        Args:
            symptoms (list): List of symptoms
            possible_diseases (list, optional): List of possible diseases
                
        Returns:
            dict: Comparison results
        """
        # Set symptoms
        self.set_symptoms(symptoms)
        
        # Perform diagnoses
        forward_results = self.diagnose_forward()
        
        # Reset knowledge base to original symptoms
        original_facts = self.kb.facts.copy()
        self.kb.facts = set()
        for symptom in symptoms:
            self.kb.add_fact(symptom, "Patient-reported symptom")
        
        backward_results = self.diagnose_backward(possible_diseases)
        
        # Restore all facts from forward reasoning for visualization
        self.kb.facts = original_facts
        
        # Display comparisons
        print("\n===== DIAGNOSTIC COMPARISON =====")
        print(f"Patient symptoms: {', '.join(symptoms)}")
        
        print("\nForward reasoning results:")
        print(f"Diagnoses: {', '.join(forward_results['diagnoses']) if forward_results['diagnoses'] else 'None'}")
        print(f"Execution time: {forward_results['execution_time']:.6f} seconds")
        
        print("\nBackward reasoning results:")
        print(f"Diagnoses: {', '.join(backward_results['diagnoses']) if backward_results['diagnoses'] else 'None'}")
        print(f"Execution time: {backward_results['execution_time']:.6f} seconds")
        
        # Visualize forward reasoning
        print("\nVisualizing forward reasoning inference chain...")
        self.forward.visualize_inference()
        
        # Visualize backward reasoning for each proven disease
        if backward_results['diagnoses']:
            for disease in backward_results['diagnoses']:
                print(f"\nVisualizing backward reasoning proof for {disease}...")
                self.backward.visualize_proof(disease)
                print("\nDetailed proof explanation:")
                print(self.backward.explain_proof(disease))
        else:
            print("\nNo diseases were proven with backward reasoning.")
            self.backward.visualize_inference()
        
        # Compare performance with a bar chart
        comparison = {
            'forward': forward_results,
            'backward': backward_results
        }
        self._visualize_comparison(comparison)
        
        return comparison
    
    def _visualize_comparison(self, comparison):
        """
        Visualize comparison between forward and backward reasoning.
        
        Args:
            comparison (dict): Comparison results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Data for plots
        approaches = ['Forward', 'Backward']
        execution_times = [
            comparison['forward']['execution_time'],
            comparison['backward']['execution_time']
        ]
        num_diagnoses = [
            len(comparison['forward']['diagnoses']),
            len(comparison['backward']['diagnoses'])
        ]
        
        # Bar chart of execution times
        ax1.bar(approaches, execution_times, color=['blue', 'green'], alpha=0.7)
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        
        # Add text labels
        for i, v in enumerate(execution_times):
            ax1.text(i, v + 0.0001, f"{v:.6f}s", ha='center')
        
        # Bar chart of number of diagnoses
        ax2.bar(approaches, num_diagnoses, color=['blue', 'green'], alpha=0.7)
        ax2.set_title('Number of Diagnoses')
        ax2.set_ylabel('Count')
        
        # Add text labels
        for i, v in enumerate(num_diagnoses):
            ax2.text(i, v + 0.1, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # If diagnoses differ, show a Venn diagram
        forward_set = set(comparison['forward']['diagnoses'])
        backward_set = set(comparison['backward']['diagnoses'])
        
        if forward_set != backward_set:
            plt.figure(figsize=(10, 6))
            
            # Create a Venn diagram
            from matplotlib_venn import venn2
            
            v = venn2([forward_set, backward_set], ('Forward Reasoning', 'Backward Reasoning'))
            
            plt.title('Comparison of Diagnoses by Reasoning Approach')
            plt.tight_layout()
            plt.show()


def run_case_study(assistant, symptoms, title):
    """
    Run a case study with the diagnostic assistant.
    
    Args:
        assistant (DiagnosticAssistant): The assistant
        symptoms (list): List of symptoms
        title (str): Title for the case study
        
    Returns:
        dict: Case study results
    """
    print(f"\n===== CASE STUDY: {title} =====")
    print(f"Patient symptoms: {', '.join(symptoms)}")
    
    # Set symptoms
    assistant.set_symptoms(symptoms)
    
    # Forward reasoning
    print("\nPerforming forward reasoning...")
    forward_results = assistant.diagnose_forward()
    
    print("Forward reasoning diagnoses:")
    if forward_results['diagnoses']:
        for diagnosis in forward_results['diagnoses']:
            print(f"- {diagnosis}")
    else:
        print("- No clear diagnosis")
    
    # Explain forward reasoning
    print("\nForward reasoning explanation:")
    print(assistant.forward.explain_inference())
    
    # Reset knowledge base to only symptoms
    kb_facts = assistant.kb.facts.copy()
    assistant.kb.facts = set()
    for symptom in symptoms:
        assistant.kb.add_fact(symptom, "Patient-reported symptom")
    
    # Backward reasoning
    print("\nPerforming backward reasoning...")
    
    # Get all possible diseases from rules
    possible_diseases = set()
    for rule in assistant.kb.rules:
        for consequent in rule.consequents:
            if not consequent.endswith("_symptom") and not consequent.startswith("test_"):
                possible_diseases.add(consequent)
    
    backward_results = assistant.diagnose_backward(possible_diseases)
    
    print("Backward reasoning diagnoses:")
    if backward_results['diagnoses']:
        for diagnosis in backward_results['diagnoses']:
            print(f"- {diagnosis}")
    else:
        print("- No clear diagnosis")
    
    # Restore all facts for visualization
    assistant.kb.facts = kb_facts
    
    # Visualizations
    print("\nVisualizing results...")
    assistant.forward.visualize_inference()
    
    # Show backward reasoning proof for diagnoses
    for diagnosis in backward_results['diagnoses']:
        assistant.backward.visualize_proof(diagnosis)
    
    # Compare approaches
    print("\nComparing reasoning approaches...")
    comparison = {
        'forward': forward_results,
        'backward': backward_results
    }
    assistant._visualize_comparison(comparison)
    
    return {
        'title': title,
        'symptoms': symptoms,
        'forward': forward_results,
        'backward': backward_results
    }


def main():
    """Main function to run the diagnostic assistant."""
    print("===== AI-based Medical Diagnostic Assistant =====")
    
    # Create diagnostic assistant
    assistant = DiagnosticAssistant()
    
    # Case studies
    case_studies = [
        {
            'title': "Common Cold vs. Allergies",
            'symptoms': ["runny_nose", "sneezing", "sore_throat"]
        },
        {
            'title': "Flu-like Symptoms",
            'symptoms': ["fever", "muscle_ache", "fatigue", "sore_throat"]
        },
        {
            'title': "Possible Respiratory Infection",
            'symptoms': ["persistent_cough", "fever", "green_phlegm", "chest_discomfort"]
        },
        {
            'title': "Severe Headache Investigation",
            'symptoms': ["headache", "nausea", "light_sensitivity"]
        }
    ]
    
    # Run each case study
    results = []
    for case in case_studies:
        result = run_case_study(assistant, case['symptoms'], case['title'])
        results.append(result)
    
    print("\n===== Medical Diagnostic Assistant Demo Complete =====")
    print("Demonstrated forward and backward reasoning for medical diagnosis")
    print("Identified different diagnoses based on reported symptoms")
    print("Visualized inference chains and reasoning processes")
    print("Compared performance and accuracy of different reasoning approaches")


if __name__ == "__main__":
    main()