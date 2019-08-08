package weka.classifiers.meta;

import weka.classifiers.CheckClassifier;
import weka.classifiers.meta.EMTransduction;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.BatchPredictor;

public class CoEMTransduction extends MultipleClassifiersCombiner {

    // private int m_NumOfClasses;

    private Instances m_labelledData;

    // private Attribute m_ClassAttribute;

    private int m_NumIterations = 10;

//    private int m_NumClassifiers;

//    private Classifier m_Classifier;

    double[][] data_distributions;

    @Override
	public void buildClassifier(Instances data) throws Exception {

	getCapabilities().testWithFail(data);

	m_labelledData = new Instances(data);

    }

    @Override
	public double[][] distributionsForInstances(Instances data) throws Exception {

	int i = 1;
	int j = m_Classifiers.length;
	int k = m_NumIterations;
	double[][] classProb = new double[data.numInstances()][data.classAttribute().numValues()];
	double[] cPro;
	Instances newData;

	m_Classifiers[0].buildClassifier(m_labelledData);

	classProb = ((BatchPredictor) m_Classifiers[0]).distributionsForInstances(data);

	while (i < k * j) {
	    newData = new Instances(m_labelledData);
	    
		for (int r = 0; r < data.numInstances(); r++) {
		    for (int l = 0; l <= m_labelledData.classAttribute().numValues() - 1; l++) {
			if (Utils.gr(classProb[r][l], 0)) {
			    Instance newInstance = (Instance) data.instance(r).copy();
			    newInstance.setClassValue(l);
			    newInstance.setWeight(classProb[r][l]);
			    newData.add(newInstance);
			}
		    }
		}
		
	    m_Classifiers[i % j].buildClassifier(newData);
	    classProb = ((BatchPredictor) m_Classifiers[i % j]).distributionsForInstances(data);
	    i++;
	}

	data_distributions = new double[data.numInstances()][data.classAttribute().numValues()];

	for (int m = 0; m <= j - 1; m++) {
	    for (int n = 0; n < data.numInstances(); n++) {
		cPro = m_Classifiers[m].distributionForInstance(data.instance(n));
		for (int p = 0; p < data.classAttribute().numValues(); p++) {
		    data_distributions[n][p] += cPro[p] / j;
		}
	    }
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
	runClassifier(new CoEMTransduction(), args);
    }

}