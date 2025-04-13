## Modified TOPSIS Calculation

This code implements the **Modified TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) method to rank features based on their importance in predicting employee turnover intention.

### Steps:
1. **Normalization**:
   The data is normalized by dividing each feature by the square root of the sum of squared values across each column, ensuring that all features are on a comparable scale.

2. **Weighted Normalized Decision Matrix**:
   After normalization, each feature is weighted using a predefined weight matrix to emphasize the importance of each feature.

3. **Ideal and Nadir Points**:
   The **ideal best** and **ideal worst** values are calculated using the median of the weighted data. These represent the most desirable and least desirable feature values.

4. **Robust Distance Calculation**:
   The distance to the ideal best and ideal worst points is calculated using three distance metrics:
   - **Manhattan Distance**
   - **Euclidean Distance**
   - **Chebyshev Distance**
   
   The distance values are combined into a **robust distance** to account for all metrics simultaneously.

5. **TOPSIS Score**:
   The **TOPSIS score** is then calculated using a weighted combination of the distances to the ideal best and worst points. The formula used here is:
   \[
   \text{TOPSIS score} = 0.5 \times \left(\frac{\text{distance to worst}}{\text{distance to best} + \text{distance to worst}}\right) + 0.5 \times \left(\frac{1}{1 + \text{distance to best}}\right)
   \]

### Dataset and Weight Matrix:
- **Dataset**: Contains 16 features related to work satisfaction, mental wellness, and demographic information.
- **Weight Matrix**: The weight matrix is set to `[0.33, 0.33, 0.33]` for each of the three features in the example dataset.

### Code Execution:
1. The **data** matrix is provided, containing the feature values (obtained using select k best, Information gain and rfe).
2. The **weights** for each feature are applied to the dataset (equal weight was used).
3. The **TOPSIS ranking** is computed for each feature, with the features being sorted by their respective TOPSIS scores.
4. The **sorted features** are printed along with their corresponding TOPSIS ranks.

