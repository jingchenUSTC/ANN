package com.jingchen.ann;

import java.util.ArrayList;
import java.util.List;

/**
 * 人工神经网络分类器
 * 
 * @author chenjing
 * 
 */
public class AnnClassifier
{
	private int mInputCount;
	private int mHiddenCount;
	private int mOutputCount;

	private List<Node> mInputNodes;
	private List<Node> mHiddenNodes;
	private List<Node> mOutputNodes;

	private float[][] mInputHiddenWeight;
	private float[][] mHiddenOutputWeight;

	private List<DataNode> trainNodes;

	public void setTrainNodes(List<DataNode> trainNodes)
	{
		this.trainNodes = trainNodes;
	}

	public AnnClassifier(int inputCount, int hiddenCount, int outputCount)
	{
		trainNodes = new ArrayList<DataNode>();
		mInputCount = inputCount;
		mHiddenCount = hiddenCount;
		mOutputCount = outputCount;
		mInputNodes = new ArrayList<Node>();
		mHiddenNodes = new ArrayList<Node>();
		mOutputNodes = new ArrayList<Node>();
		mInputHiddenWeight = new float[inputCount][hiddenCount];
		mHiddenOutputWeight = new float[mHiddenCount][mOutputCount];
	}

	/**
	 * 更新权重
	 */
	private void updateWeights(float eta)
	{
		for (int i = 0; i < mInputCount; i++)
			for (int j = 0; j < mHiddenCount; j++)
				mInputHiddenWeight[i][j] -= eta
						* mInputNodes.get(i).getForwardOutputValue()
						* mHiddenNodes.get(j).getBackwardOutputValue();
		for (int i = 0; i < mHiddenCount; i++)
			for (int j = 0; j < mOutputCount; j++)
				mHiddenOutputWeight[i][j] -= eta
						* mHiddenNodes.get(i).getForwardOutputValue()
						* mOutputNodes.get(j).getBackwardOutputValue();
	}

	/**
	 * 前向传播
	 */
	private void forwrad(List<Float> list)
	{
		// 输入层
		for (int k = 0; k < list.size(); k++)
			mInputNodes.get(k).setForwardInputValue(list.get(k));
		// 隐层
		for (int j = 0; j < mHiddenCount; j++)
		{
			float temp = 0;
			for (int k = 0; k < mInputCount; k++)
				temp += mInputHiddenWeight[k][j]
						* mInputNodes.get(k).getForwardOutputValue();
			mHiddenNodes.get(j).setForwardInputValue(temp);
		}
		// 输出层
		for (int j = 0; j < mOutputCount; j++)
		{
			float temp = 0;
			for (int k = 0; k < mHiddenCount; k++)
				temp += mHiddenOutputWeight[k][j]
						* mHiddenNodes.get(k).getForwardOutputValue();
			mOutputNodes.get(j).setForwardInputValue(temp);
		}
	}

	/**
	 * 反向传播
	 */
	private void backward(int type)
	{
		// 输出层
		for (int j = 0; j < mOutputCount; j++)
		{
			float result = -1;
			if (j == type)
				result = 1;
			mOutputNodes.get(j).setBackwardInputValue(
					mOutputNodes.get(j).getForwardOutputValue() - result);
		}
		// 隐层
		for (int j = 0; j < mHiddenCount; j++)
		{
			float temp = 0;
			for (int k = 0; k < mOutputCount; k++)
				temp += mHiddenOutputWeight[j][k]
						* mOutputNodes.get(k).getBackwardOutputValue();
		}
	}

	public void train(float eta, int n)
	{
		reset();
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < trainNodes.size(); j++)
			{
				forwrad(trainNodes.get(j).getAttribList());
				backward(trainNodes.get(j).getType());
				updateWeights(eta);
			}

		}
	}

	private void reset()
	{
		for (int i = 0; i < mInputCount; i++)
			mInputNodes.add(new Node(Node.TYPE_INPUT));
		for (int i = 0; i < mHiddenCount; i++)
			mHiddenNodes.add(new Node(Node.TYPE_HIDDEN));
		for (int i = 0; i < mOutputCount; i++)
			mOutputNodes.add(new Node(Node.TYPE_OUTPUT));
		for (int i = 0; i < mInputCount; i++)
			for (int j = 0; j < mHiddenCount; j++)
				mInputHiddenWeight[i][j] = (float) (Math.random() * 0.1);
		for (int i = 0; i < mHiddenCount; i++)
			for (int j = 0; j < mOutputCount; j++)
				mHiddenOutputWeight[i][j] = (float) (Math.random() * 0.1);
	}

	public int test(DataNode dn)
	{
		forwrad(dn.getAttribList());
		float result = 2;
		int type = 0;
		for (int i = 0; i < mOutputCount; i++)
			if ((1 - mOutputNodes.get(i).getForwardOutputValue()) < result)
			{
				result = 1 - mOutputNodes.get(i).getForwardOutputValue();
				type = i;
			}
		return type;
	}
}
