import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create some 3D data (for example, a 3D line plot)
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [5, 6, 7, 8, 9]

# Create a 3D line plot
ax.plot(x, y, z, c='b', marker='o')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the 3D plot
plt.show()


