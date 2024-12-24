#!/usr/bin/env python3
"""test_cluster.py - unit tests for the BattleClustering class"""
import os
import unittest

import pandas as pd

from battle_clustering import BattleClustering


class TestBattleClustering(unittest.TestCase):
    """Unit tests for the BattleClustering class"""
    @classmethod
    def setUpClass(cls):
        # Create a sample dataset for testing
        cls.sample_data = pd.DataFrame({
            "Battle_Name": ["Austerlitz", "Waterloo", "Borodino"],
            "Terrain": ["Hills, Lakes", "Flat Plains", "Flat, River Nearby"],
            "Weather": ["Foggy", "Rainy", "Clear"],
            "Key_Factors": ["Flanking", "Direct Assault", "Artillery"],
            "Outcome": ["Victory", "Defeat", "Stalemate"]
        })
        cls.sample_data_path = "sample_battles.csv"
        cls.sample_data.to_csv(cls.sample_data_path, index=False)
        cls.output_path = "test_output.csv"
        cls.visualization_path = "test_visualization.png"

    @classmethod
    def tearDownClass(cls):
        # Clean up the sample data file
        os.remove(cls.sample_data_path)
        if os.path.exists(cls.output_path):
            os.remove(cls.output_path)
        if os.path.exists(cls.visualization_path):
            os.remove(cls.visualization_path)

    def setUp(self):
        self.clustering = BattleClustering(data_path=self.sample_data_path)

    def test_load_data(self):
        """Test the load_data method"""
        self.clustering.load_data()
        self.assertIsNotNone(self.clustering.data)
        self.assertEqual(len(self.clustering.data), 3)

    def test_preprocess_data(self):
        """Test the preprocess_data method"""
        self.clustering.load_data()
        self.clustering.preprocess_data()
        self.assertIsNotNone(self.clustering.X_scaled)
        self.assertEqual(self.clustering.X_scaled.shape[0], 3)

    def test_perform_clustering(self):
        """Test the perform_clustering method"""
        self.clustering.load_data()
        self.clustering.preprocess_data()
        self.clustering.perform_clustering()
        self.assertIsNotNone(self.clustering.clusters)
        self.assertEqual(len(self.clustering.clusters), 3)

    def test_save_results(self):
        """Test the save_results method
        """
        self.clustering.load_data()
        self.clustering.preprocess_data()
        self.clustering.perform_clustering()
        self.clustering.save_results(output_path=self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_visualize_clusters(self):
        self.clustering.load_data()
        self.clustering.preprocess_data()
        self.clustering.perform_clustering()
        self.clustering.visualize_clusters(output_path=self.visualization_path)
        self.assertTrue(os.path.exists(self.visualization_path))

if __name__ == "__main__":
    unittest.main()