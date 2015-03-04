package com.jingchen.ann;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;

/**
 * 说明：目前使用的这份测试集是从原始数据中随机抽取26个组成的
 * 
 * @author chenjing
 * 
 */
public class MainClass
{
	public static void main(String[] args) throws Exception
	{
		DataUtil util = DataUtil.getInstance();
		List<DataNode> trainList = util.getDataList("E:/train.txt");
		List<DataNode> testList = util.getDataList("E:/test.txt");
		BufferedWriter output = new BufferedWriter(new FileWriter(new File(
				"E:/annoutput.txt")));
		int typeCount = util.getTypeCount();
		AnnClassifier annClassifier = new AnnClassifier(trainList.get(0)
				.getAttribList().size(), 10, typeCount);
		annClassifier.setTrainNodes(trainList);
		annClassifier.train(0.5f, 5000);
		for (int i = 0; i < testList.size(); i++)
		{
			DataNode test = testList.get(i);
			int type = annClassifier.test(test);
			List<Float> attribs = test.getAttribList();
			for (int n = 0; n < attribs.size(); n++)
			{
				output.write(attribs.get(n) + ",");
				output.flush();
			}
			output.write(util.getTypeName(type) + "\n");
			output.flush();
		}
		output.close();

	}

}
