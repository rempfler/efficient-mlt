These are the instances used in the publications:

Jug F., Levinkov E., Blasse C., Myers E. W. and Andres B. "Moral Lineage Tracing", CVPR 2016

and

Rempfler M., Lange J.-H., Jug F., Blasse C., Myers E. W., Menze B. H., and Andres B., "Efficient algorithms for moral lineage tracing", ICCV 2017, pp. 4695-4704


Each dataset consists of two files describing nodes and edges of the graph.

nodes.csv constains information about each vertex per line in the following format:

frame_number node_id_in_frame x y p
frame_number node_id_in_frame x y p
frame_number node_id_in_frame x y p
frame_number node_id_in_frame x y p

where x and y are coordinates of the center of mass of the supervoxel in frame and p is the probability of the superpixel to be born or terminated in the current frame. The latter is used in the presented experiments.


edges.csv contains information about each edge per line in the following format:

frame_number_0 node_id_0 frame_number_1 node_id_1 p
frame_number_0 node_id_0 frame_number_1 node_id_1 p
frame_number_0 node_id_0 frame_number_1 node_id_1 p
frame_number_0 node_id_0 frame_number_1 node_id_1 p

frame_number and node_id strictly coincides with te indexing defined in nodes.csv. Here, p is the estimated probability of an edge being cut.

To easily load the data, just use the code in lineage/problem.hxx
