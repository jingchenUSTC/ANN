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

	private List<NetworkNode> mInputNodes;
	private List<NetworkNode> mHiddenNodes;
	private List<NetworkNode> mOutputNodes;

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
		mInputNodes = new ArrayList<NetworkNode>();
		mHiddenNodes = new ArrayList<NetworkNode>();
		mOutputNodes = new ArrayList<NetworkNode>();
		mInputHiddenWeight = new float[inputCount][hiddenCount];
		mHiddenOutputWeight = new float[mHiddenCount][mOutputCount];
	}

	/**
	 * 更新权重，每个权重的梯度都等于与其相连的前一层节点的输出乘以与其相连的后一层的反向传播的输出
	 */
	private void updateWeights(float eta)
	{
		// 更新输入层到隐层的权重矩阵
		for (int i = 0; i < mInputCount; i++)
			for (int j = 0; j < mHiddenCount; j++)
				mInputHiddenWeight[i][j] -= eta
						* mInputNodes.get(i).getForwardOutputValue()
						* mHiddenNodes.get(j).getBackwardOutputValue();
		// 更新隐层到输出层的权重矩阵
		for (int i = 0; i < mHiddenCount; i++)
			for (int j = 0; j < mOutputCount; j++)
				mHiddenOutputWeight[i][j] -= eta
						* mHiddenNodes.get(i).getForwardOutputValue()
						* mOutputNodes.get(j).getBackwardOutputValue();
	}

	/**
	 * 前向传播
	 */
	private void forward(List<Float> list)
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
			// 输出层计算误差把误差反向传播，这里-1代表不属于，1代表属于
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
				forward(trainNodes.get(j).getAttribList());
				backward(trainNodes.get(j).getType());
				updateWeights(eta);
			}

		}
	}

	/**
	 * 初始化
	 */
	private void reset()
	{
		mInputNodes.clear();
		mHiddenNodes.clear();
		mOutputNodes.clear();
		for (int i = 0; i < mInputCount; i++)
			mInputNodes.add(new NetworkNode(NetworkNode.TYPE_INPUT));
		for (int i = 0; i < mHiddenCount; i++)
			mHiddenNodes.add(new NetworkNode(NetworkNode.TYPE_HIDDEN));
		for (int i = 0; i < mOutputCount; i++)
			mOutputNodes.add(new NetworkNode(NetworkNode.TYPE_OUTPUT));
		for (int i = 0; i < mInputCount; i++)
			for (int j = 0; j < mHiddenCount; j++)
				mInputHiddenWeight[i][j] = (float) (Math.random() * 0.1);
		for (int i = 0; i < mHiddenCount; i++)
			for (int j = 0; j < mOutputCount; j++)
				mHiddenOutputWeight[i][j] = (float) (Math.random() * 0.1);
	}

	public int test(DataNode dn)
	{
		forward(dn.getAttribList());
		float result = 2;
		int type = 0;
		// 取最接近1的
		for (int i = 0; i < mOutputCount; i++)
			if ((1 - mOutputNodes.get(i).getForwardOutputValue()) < result)
			{
				result = 1 - mOutputNodes.get(i).getForwardOutputValue();
				type = i;
			}
		return type;
	}
}
