package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.Copyable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class EMTransduction extends IteratedSingleClassifierEnhancer {

	private Instances m_labelledData;


	double[][] data_distributions;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		getCapabilities().testWithFail(data);

		m_labelledData = new Instances(data);

	}

	@Override
	public double[][] distributionsForInstances(Instances data) throws Exception {

		int i = getNumIterations();
		int j = 1;
		double[] classProb;
		Instances newData;

		
		m_Classifier.buildClassifier(m_labelledData);

		while (j < i) {
			newData = new Instances(m_labelledData);
			for (Instance instance : data) {
				classProb = m_Classifier.distributionForInstance(instance);
				for (int k = 0; k <= m_labelledData.classAttribute().numValues() - 1; k++) {
					if (Utils.gr(classProb[k], 0)) {
						Instance newInstance = (Instance) instance.copy();
						newInstance.setClassValue(k);
						newInstance.setWeight(classProb[k]);
						newData.add(newInstance);
					}
				}
			}
			m_Classifier.buildClassifier(newData);
			j++;
		}
		data_distributions = new double[data.numInstances()][data.classAttribute().numValues()];
		for (int l = 0; l < data.numInstances(); l++) {
			double[] clProFoIn = m_Classifier.distributionForInstance(data.instance(l));
			for (int m = 0; m < data.classAttribute().numValues(); m++) {
			data_distributions[l][m] = clProFoIn[m];}
			
		}
		return data_distributions;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws UnsupportedOperationException {
		return null;

	}

	@Override
	public boolean implementsMoreEfficientBatchPrediction() {
		return true;

	}

	@Override
	public String toString() {
		StringBuilder text = new StringBuilder();
		return text.toString();
	}

	public static void main(String[] args) {
		
		runClassifier(new EMTransduction(), args);
	}
}
