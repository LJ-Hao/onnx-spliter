import onnx
import argparse
import json

def split_onnx_graph(model_path, nodes_to_split):
    # 加载ONNX模型
    model = onnx.load(model_path)
    
    # 获取模型的图
    graph = model.graph
    
    # 获取所有节点名称
    node_names = [node.name for node in graph.node]
    
    # 存储结果
    results = {}

    # 遍历每个分割节点
    for split_node_name in nodes_to_split:
        split_index = None
        
        # 查找拆分节点的索引
        for idx, node_name in enumerate(node_names):
            if node_name == split_node_name:
                split_index = idx
                break
        
        if split_index is None:
            raise ValueError(f"Node {split_node_name} not found in the graph.")
        
        # 获取分割节点及其输出
        split_node = graph.node[split_index]
        split_node_outputs = split_node.output  # 获取所有输出节点
        
        # 获取输出形状
        output_shapes = {}
        for output in split_node_outputs:
            for value_info in graph.value_info:
                if value_info.name == output:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    output_shapes[output] = shape

        # Part 1: 包含分割节点之前的所有节点
        part1_nodes = graph.node[:split_index + 1]
        part1_inputs = graph.input[:]  # part1的输入是原始模型的所有输入
        part1_outputs = [onnx.ValueInfoProto(name=output) for output in split_node_outputs]

        # Part 2: 从分割节点之后的节点开始
        part2_nodes = graph.node[split_index + 1:]
        part2_inputs = [onnx.ValueInfoProto(name=output) for output in split_node_outputs]
        part2_outputs = graph.output[:]  # part2的输出是原始模型的所有输出

        # 创建 part1 的模型（添加输出节点）
        part1_graph = onnx.GraphProto(
            node=part1_nodes,
            input=part1_inputs,
            output=part1_outputs
        )
        part1_model = onnx.ModelProto(graph=part1_graph)

        # 创建 part2 的模型（添加输入节点）
        part2_graph = onnx.GraphProto(
            node=part2_nodes,
            input=part2_inputs,
            output=part2_outputs
        )
        part2_model = onnx.ModelProto(graph=part2_graph)

        # 保存 part1 和 part2 的模型
        part1_path = f"part1.onnx"
        part2_path = f"part2.onnx"
        onnx.save(part1_model, part1_path)
        onnx.save(part2_model, part2_path)

        # 将结果存储到字典中
        results[split_node_name] = {
            "part1_path": part1_path,
            "part2_path": part2_path,
            "output_shapes": output_shapes
        }
    
    # 返回结果作为JSON对象
    print(results)
    return json.dumps(results, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Split ONNX model into parts from specified nodes")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--node", type=str, required=True, help="Comma-separated list of node names to split on")
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析节点列表
    nodes_to_split = args.node.split(",")
    
    # 调用函数拆分ONNX图
    result_json = split_onnx_graph(args.model, nodes_to_split)
    print(result_json)


if __name__ == "__main__":
    main()
