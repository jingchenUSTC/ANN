package com.jingchen.ann;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;

import com.jingchen.util.ConsoleHelper;
import com.jingchen.util.DataUtil;

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
		if (args.length < 5)
		{
			System.out
					.println("Usage: \n\t-train trainfile\n\t-test predictfile\n\t-sep separator, default:','\n\t-eta eta, default:0.5\n\t-iter iternum, default:5000\n\t-out outputfile");
			return;
		}
		ConsoleHelper helper = new ConsoleHelper(args);
		String trainfile = helper.getArg("-train", "");
		String testfile = helper.getArg("-test", "");
		String separator = helper.getArg("-sep", ",");
		String outputfile = helper.getArg("-out", "");
		float eta = helper.getArg("-eta", 0.5f);
		int nIter = helper.getArg("-iter", 5000);
		DataUtil util = DataUtil.getInstance();
		List<DataNode> trainList = util.getDataList(trainfile, separator);
		List<DataNode> testList = util.getDataList(testfile, separator);
		BufferedWriter output = new BufferedWriter(new FileWriter(new File(
				outputfile)));
		int typeCount = util.getTypeCount();
		AnnClassifier annClassifier = new AnnClassifier(trainList.get(0)
				.getAttribList().size(), 10, typeCount);
		annClassifier.setTrainNodes(trainList);
		annClassifier.train(eta, nIter);
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
