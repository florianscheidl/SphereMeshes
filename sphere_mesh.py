import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits

from icosphere import icosphere
from itertools import combinations
from tqdm import tqdm


class SphereMesh:

    def __init__(self, type: str = "icosphere", level: int = 1, return_format : str = None):

        # vertices and faces
        self.vertices, self.faces = icosphere(nu=level)
        # self.faces = [[self.vertices[point] for point in face] for face in self.faces] # use coordinates instead of indices

        # cast to list
        self.vertices = list(self.vertices)
        self.faces = list(self.faces)

        if type == "goldberg_polyhedron":
            self.vertices, self.faces = self.build_dual(self.vertices, self.faces)
        if type == "icosphere":
            tmp_vertices, tmp_faces = self.build_dual(self.vertices, self.faces)
            self.vertices, self.faces = self.build_dual(tmp_vertices, tmp_faces)

        # face areas
        self.face_areas = []
        for face in self.faces:
            face_vertices = [self.vertices[j] for j in face]
            self.face_areas.append(poly_area(face_vertices))

        # edges:
        self.edges_by_face_indices = []
        self.edges_by_vertex_indices = []
        for i, j in combinations(range(len(self.faces)), 2):
            face_intersection = set([i for i in set.intersection(set(self.faces[i]), set(self.faces[j])) if i!=0])
            if len(face_intersection) == 2:
                self.edges_by_face_indices.append([i, j])  # could potentially use set here instead ...
                # edges_by_face_indices.append({i,j}) # set here instead ...
                self.edges_by_vertex_indices.append(list(face_intersection))

        # after all operations, we return as array
        if return_format == "as_array":
            self.face_areas = np.array(self.face_areas)
            self.vertices = np.array(self.vertices)
            self.faces = np.array(self.faces)
            self.edges_by_face_indices = np.array(self.edges_by_face_indices)
            self.edges_by_vertex_indices = np.array(self.edges_by_vertex_indices)

    def build_dual(self, vertices, faces):
        """
        Given a geodesic polyhedron (as vertices and faces), computes the dual (i.e. a Goldberg polyhedron). The dual operation build a new set
        of vertices and faces as follows. A vertex in the dual is formed by the centers of each face in the original
        polyhedron. The faces of the dual are formed by connecting each set of new vertices for which the corresponding
        faces touched in the original polyhedron
        :return: vertices (coordinates), faces (indices of vertices they connect)

        :param vertices: list of three-dimensional coordinates
        :param faces: indices of vertices
        :return: new lists of vertices and faces
        """

        new_vertices = [compute_midpoint(vertices, face) for face in tqdm(faces)]

        new_faces = []
        # for v_indx in tqdm(range(len(vertices))):
        for i, vertex in enumerate(vertices):
            # new_face = [face for face in faces if vertex in [vertices[point] for point in face]]
            new_face = []
            for j, face in enumerate(faces):
                if i in face:
                    new_face.append(j)
                        # new_face.append([i for i in range(len(faces)) if (np.array(faces)[i,:]==face).all()][0])

            new_faces.append(new_face)

        return new_vertices, new_faces

    def visualise(self, show_midpoints: bool = False):

        fig = plt.figure(figsize=(15, 10))

        # creating mesh
        # poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(self.vertices[self.faces])
        poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([np.array([self.vertices[v_idx] for v_idx in face]) for face in self.faces])
        poly.set_edgecolor('black')
        poly.set_linewidth(0.25)

        # would be nice to visualise edges (to make sure what we defined as edges is the right thing)

        # and now -- visualization!
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        ax.add_collection3d(poly)

        # if show_midpoints:
        #     midpoints = compute_midpoint(self.vertices, self.faces)

            # ax.scatter(midpoints[:,0],midpoints[:,1],midpoints[:,2], marker = "o", color="orange", s=40)
        ax.scatter(np.array(self.vertices)[:,0],np.array(self.vertices)[:,1],np.array(self.vertices)[:,2], marker = "o", color="green", s=40)

        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)

        # ax.set_xticks([-1, 0, 1])
        # ax.set_yticks([-1, 0, 1])
        # ax.set_zticks([-1, 0, 1])

        # ax.set_title(f'Title TBA')

        fig.suptitle('Icospheres with different subdivision frequency')
        plt.show()


# copied from https://code.activestate.com/recipes/578276-3d-polygon-area/
def poly_area(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i + 1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
    return (x / magnitude, y / magnitude, z / magnitude)

def compute_midpoint(vertices, face, show: bool = True):

    adj_vertices = np.array([vertices[indx] for indx in face])
    midpoint = list(np.mean(adj_vertices, axis=0))

        # if show:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.scatter(midpoint[0],midpoint[1],midpoint[2])
        #     ax.scatter(adj_vertices[:,0],adj_vertices[:,1],adj_vertices[:,2])
        #     plt.show()

    return midpoint


if __name__ == "__main__":
    mesh = SphereMesh(level=1)
    # mesh = SphereMesh(type="goldberg_polyhedron", level=1)
    mesh.visualise(show_midpoints=True)

    print("Done.")
    # mesh.visualise(show_midpoints=True)
