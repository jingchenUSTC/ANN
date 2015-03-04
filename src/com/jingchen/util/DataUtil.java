package com.jingchen.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.jingchen.ann.DataNode;

public class DataUtil
{
	private static DataUtil instance = null;
	private Map<String, Integer> mTypes;
	private int mTypeCount;

	private DataUtil()
	{
		mTypes = new HashMap<String, Integer>();
		mTypeCount = 0;
	}

	public static synchronized DataUtil getInstance()
	{
		if (instance == null)
			instance = new DataUtil();
		return instance;

	}

	public Map<String, Integer> getTypeMap()
	{
		return mTypes;
	}

	public int getTypeCount()
	{
		return mTypeCount;
	}

	public String getTypeName(int type)
	{
		if (type == -1)
			return new String("无法判断");
		Iterator<String> keys = mTypes.keySet().iterator();
		while (keys.hasNext())
		{
			String key = keys.next();
			if (mTypes.get(key) == type)
				return key;
		}
		return null;
	}

	/**
	 * 根据文件生成训练集，注意：程序将以第一个出现的非数字的属性作为类别名称
	 * 
	 * @param fileName
	 *            文件名
	 * @param sep
	 *            分隔符
	 * @return
	 * @throws Exception
	 */
	public List<DataNode> getDataList(String fileName, String sep)
			throws Exception
	{
		List<DataNode> list = new ArrayList<DataNode>();
		BufferedReader br = new BufferedReader(new FileReader(
				new File(fileName)));
		String line = null;
		while ((line = br.readLine()) != null)
		{
			String splits[] = line.split(sep);
			DataNode node = new DataNode();
			int i = 0;
			for (; i < splits.length; i++)
			{
				try
				{
					node.addAttrib(Float.valueOf(splits[i]));
				} catch (NumberFormatException e)
				{
					// 非数字，则为类别名称，将类别映射为数字
					if (!mTypes.containsKey(splits[i]))
					{
						mTypes.put(splits[i], mTypeCount);
						mTypeCount++;
					}
					node.setType(mTypes.get(splits[i]));
					list.add(node);
				}
			}
		}
		return list;
	}
}
