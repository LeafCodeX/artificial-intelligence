import numpy as np

from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic

def main():
    np.random.seed(123)

    train_data, test_data = load_titanic()

    dt = DecisionTree({"depth": 14})
    print("==========================")
    print("[INFO] Decision Tree:")
    dt.train(*train_data)
    dt.evaluate(*train_data)
    print("[INFO] Decision Tree Test:")
    dt.evaluate(*test_data)

    rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 14})
    print("==========================")
    print("[INFO] Random Forest:")
    rf.train(*train_data)
    rf.evaluate(*train_data)
    print("[INFO] Random Forest Test:")
    rf.evaluate(*test_data)
    print("==========================")

if __name__=="__main__":
    main()