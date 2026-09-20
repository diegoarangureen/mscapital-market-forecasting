import numpy as np


def cosine_similarity_score(true_values, predicted_values):
    """Calculate cosine similarity between two one-dimensional arrays."""

    # TODO 1：使用 np.dot() 计算点积。
    # TODO 1: Calculate the dot product with np.dot().
    dot_result=np.dot(true_values, predicted_values)
    # TODO 2：分别使用 np.linalg.norm() 计算两个向量的长度。
    # TODO 2: Calculate both vector lengths with np.linalg.norm().
    norm_result=np.linalg.norm(true_values)*np.linalg.norm(predicted_values)

    # TODO 3：按照余弦相似度公式计算 score，并返回它。
    # TODO 3: Calculate score with the cosine-similarity formula and return it.
    score=dot_result/norm_result
    return score

# 以下数据只用于理解指标，不是比赛训练数据。
# The following data is only for understanding the metric, not competition training data.
true_values = np.array([1.0, 2.0])
same_direction = np.array([2.0, 4.0])
imperfect_direction = np.array([2.0, 1.0])
opposite_direction = np.array([-1.0, -2.0])
scaled_direction = np.array([20.0, 40.0])
zero_prediction = np.array([0.0, 0.0])

# TODO 4：调用函数并打印前四种预测的得分。
# TODO 4: Call the function and print the scores of the first four predictions.
same_score=cosine_similarity_score(true_values, same_direction)
imperfect_score=cosine_similarity_score(true_values, imperfect_direction)
opposite_score=cosine_similarity_score(true_values, opposite_direction)
scaled_score=cosine_similarity_score(true_values, scaled_direction)
print(same_score)
print(imperfect_score)
print(opposite_score)
print(scaled_score)
# TODO 5：打印 zero_prediction 的长度，不要用它计算分数。
# TODO 5: Print the length of zero_prediction; do not calculate a score with it.
print(np.linalg.norm(zero_prediction))
