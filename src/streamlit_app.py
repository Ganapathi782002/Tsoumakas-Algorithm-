import sys
import os
import pandas as pd
import streamlit as st
import time
import mydatasets
import utils
from labeltree import LabelTree, LeafSizeStoppingCondition
from partition import RandomPartitioner, KMeansPartitioner, BalancedKMeansPartitioner
from classifiers import OVAMultiLabelClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Tsoumakas2008Experiment:

    def __init__(self, dataset_name, splits, num_repititions):
        assert(dataset_name in ["mediamill", "bibtex"])
        assert(num_repititions > 0)
        # self.outfilename = outfilename
        self.num_repititions = num_repititions
        self.dataset_name = dataset_name
        self.splits = splits
        # load data
        curr_dir = os.getcwd()
        os.chdir("..")
        self.dataset, self.trn_splits, self.tst_splits = mydatasets.load_dataset(
            self.dataset_name)
        os.chdir(curr_dir)
        # static label tree arguments
        base = GaussianNB() if dataset_name == "mediamill" else BernoulliNB()
        self.static_args = {
            'stopping_condition': LeafSizeStoppingCondition(1),
            'leaf_classifier': OVAMultiLabelClassifier(base),
            'internal_classifier': OVAMultiLabelClassifier(base)
        }
        # partitioning methods
        self.methods = [RandomPartitioner,
                        KMeansPartitioner, BalancedKMeansPartitioner]
        self.results = {'RandomPartitioner': [], 'KMeansPartitioner': [], 'BalancedKMeansPartitioner': []}
        
    def run_experiment(self):
        for split_num in self.splits:
            for method in self.methods:
                for num_partitions in range(2, 9):
                    for rep_num in range(0, self.num_repititions):
                        partitioner = method(num_partitions)
                        trn_data, tst_data = mydatasets.get_small_dataset_split(
                            self.dataset, self.trn_splits, self.tst_splits, split_num)
                        x_trn, y_trn = mydatasets.get_arrays(trn_data)
                        x_tst, y_tst = mydatasets.get_arrays(tst_data)
                        # columns of y_mat are label representation in Tsoumakas08
                        repre = y_trn.T
                        row_name = str(method.__name__)+"_s="+str(split_num) + \
                            "_k="+str(num_partitions)+"/r="+str(rep_num)
                        st.write(row_name)  # Print the method name, split_number, partition number, label representation number.
                        metrics = self.train_and_test(
                            repre, x_trn, y_trn, x_tst, y_tst, partitioner)
                        self.results[method.__name__].append(metrics)
                        # Display metrics on Streamlit
                        st.subheader(row_name)
                        df_metrics = pd.DataFrame(metrics['metrics'], index=[0])
                        st.dataframe(df_metrics)

        # st.header('Evaluation Metrics')
        # if results:
        #     for method, metrics_list in results.items():
        #         st.subheader(f'{method} Metrics')
        #         df_metrics = pd.DataFrame(metrics_list)
        #         df_metrics.columns.name = 'Metrics'
        #         st.dataframe(df_metrics)
        #     self.plot_results(results)
        # else:
        #     st.warning('No evaluation metrics available.')

    def plot_results(self, results):
        methods = list(results.keys())
        avg_accuracy = [sum([metrics['metrics']['accuracy'] for metrics in result]) / len(result) for result in results.values()]

        # Extract scalar values from avg_accuracy list
        avg_accuracy_scalar = [acc[0] for acc in avg_accuracy]

        fig = go.Figure(data=[go.Bar(x=methods, y=avg_accuracy_scalar, marker_color=['blue', 'orange', 'green'])])
        fig.update_layout(
            title='Comparison of Partitioning Methods',
            xaxis_title='Partitioning Method',
            yaxis_title='Average Accuracy',
            hovermode='closest',
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
        )
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        st.plotly_chart(fig)


    def train_and_test(self, repre, x_trn, y_trn, x_tst, y_tst, partitioner):
        # init
        ltree = LabelTree(partitioner=partitioner, **self.static_args)
        ltree.num_features = x_trn.shape[1]
        ltree.num_labels = y_trn.shape[1]
        ltree.num_trn_points = x_trn.shape[0]
        # build tree
        part_time = time.perf_counter()
        ltree._fit_tree(x_trn, y_trn, repre)
        part_time = time.perf_counter() - part_time
        # train classifiers
        trn_time = time.perf_counter()
        ltree._fit_classifiers(x_trn, y_trn)
        trn_time = time.perf_counter() - trn_time
        # predict on test data
        tst_time = time.perf_counter()
        y_pred, probs_pred = ltree.predict(x_tst, threshold=0.5, method="recursive",
                                           num_paths=None, recurse_threshold=0.5, return_probs=True)
        tst_time = time.perf_counter() - tst_time
        # get performance metrics
        accuracy = accuracy_score(y_tst, y_pred)
        metrics = utils.calculate_performance_metrics(
            y_tst, y_pred, probs_pred)
        # add time metrics
        metrics['partition_time'] = part_time
        metrics['train_time'] = trn_time
        metrics['test_time'] = tst_time
        metrics['accuracy'] = accuracy
        return {'metrics': metrics}


def main():
    st.title('Tsoumakas Experiment')

    dataset_name = st.text_input('Dataset Name (mediamill or bibtex):')
    num_repititions = st.number_input(
        'Number of Repetitions (positive integer):', min_value=1, value=1)
    splits_str = st.text_input(
        'Splits (comma-separated list of integers):')
    splits = [int(x.strip()) for x in splits_str.split(',') if x.strip()]
    
    if not splits:
        st.warning('Please provide splits.')
    elif not 1 <= min(splits) <= max(splits) <= 9:
        st.warning('Splits should be between 1 and 9.')

    # outfilename = st.text_input('Output Filename:')
    
    if st.button('Run Experiment'):
        if not dataset_name or not splits:
            st.warning('Please provide all the required inputs.')
        else:
            st.info('Starting Experiment...')
            experiment = Tsoumakas2008Experiment(
                dataset_name, splits, num_repititions)
            experiment.run_experiment()
            st.success('Experiment completed successfully!')
            st.button('End Experiment')

            # Plot results after the experiment is completed
            st.info('Plotting results...')
            experiment.plot_results(experiment.results)  # Assuming you have a results attribute in Tsoumakas2008Experiment class

if __name__ == "__main__":
    main()
