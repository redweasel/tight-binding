import numpy as np
from scipy.spatial import KDTree

# My own symmetries.
# The following has a lot of it, but is missing a part which I deemed crucial for the performance.
# https://qsymm.readthedocs.io/en/latest/tutorial/bloch_generator.html

def pointcloud_distance(pointcloud1, pointcloud2):
    pointcloud1 = np.asarray(pointcloud1)
    pointcloud1 = pointcloud1.reshape(pointcloud1.shape[0], -1)
    pointcloud2 = np.asarray(pointcloud2)
    pointcloud2 = pointcloud2.reshape(pointcloud2.shape[0], -1)
    # for each point in pointcloud1 find the closest in pointcloud2, add the distance, then remove that
    dist = 0.0
    for i, p1 in enumerate(pointcloud1):
        d = np.linalg.norm(p1 - pointcloud2, axis=-1)
        min_index = np.argmin(d.flat)
        dist += d.flat[min_index]
        d = np.delete(d.flat, min_index)
    return dist

# a function that returns a function that maps positions to (neighbor_index, is_mirrored)
# and raises a ValueError if the neighbor isn't found.
def neighbor_function(neighbors):
    kdtree = KDTree(neighbors)
    err = 1e-4 # global error for symmetry points
    def find(r):
        mirror = False
        _dist, j = kdtree.query(r, distance_upper_bound=err)
        if j < 0 or j >= len(neighbors):
            _dist, j = kdtree.query(-r, distance_upper_bound=err)
            mirror = True
            if not 0 <= j < len(neighbors):
                raise ValueError("neighbors don't match the symmetry")
        return j, mirror
    return find

# a function that returns a function that maps positions to (neighbor_index, is_mirrored)
# and returns None, None if the neighbor isn't found
def try_neighbor_function(neighbors):
    kdtree = KDTree(neighbors)
    err = 1e-4 # global error for symmetry points
    def find(r):
        mirror = False
        _dist, j = kdtree.query(r, distance_upper_bound=err)
        if j < 0 or j >= len(neighbors):
            _dist, j = kdtree.query(-r, distance_upper_bound=err)
            mirror = True
            if not 0 <= j < len(neighbors):
                return None, None
        return j, mirror
    return find
    

# class for symmetries. All symmetries (except inversion symmetry) are saved as unitary/orthogonal transformation matrices
# TODO do they need to be orthogonal? Why not SL(3) or -SL(3)? Those would also create closed sets of symmetries.
class Symmetry:
    # initialize symmetry from orthogonal matrices
    def __init__(self, S, inversion=False):
        self.S = np.asarray(S)
        self.inversion = inversion
        if np.linalg.norm(self.S[0] - np.eye(len(self.S[0]))) > 1e-7:
            raise ValueError("The first entry in the symmetry list needs to be the identity element")
        if [s for s in self.S if abs(abs(np.linalg.det(s)) - 1) > 1e-7]:
            raise ValueError("invalid matrix in parameter. All matrices need to have |det| = 1, otherwise the group is infinite")
        # check if S has inversion symmetry build in
        if pointcloud_distance(self.S, -self.S) < 1e-6:
            #if not inversion:
            #    print("found inversion symmetry")
            if len(S[0]) % 2 == 1:
                # keep only the S with positive determinants (works for real matrices in odd dimensions)
                self.inversion = True
                self.S = np.array([s for s in self.S if np.linalg.det(s) > 0])
            else:
                #self.S = np.array([s for s in self.S if next((x for x in np.ravel(s) if abs(x) > 1e-4), 1) > 0])
                # here inversion can be a problem, as it may be in the center, but it's not always a normal subgroup!
                # for now, disable the inversion symmetry reduction in even dimensions.
                pass
        # TODO add lists for broken symmetries
        # each symmetry(except for inversion) has a line or a plane on which it's unbroken
        # -> add information about that and add a way to get all the unbroken/broken symmetries for a point
    
    def copy(self):
        return Symmetry(np.array(self.S), self.inversion)

    def dim(self):
        return len(self.S[0])
    
    def check(self) -> bool:
        gen_sym = Symmetry.from_generator(self.S, self.inversion)
        return self == gen_sym
    
    def __eq__(self, other):
        if self.dim() != other.dim():
            return False
        if self.inversion != other.inversion:
            return False
        if len(self) != len(other):
            return False
        return pointcloud_distance(self.S, other.S) < len(self.S) * 1e-7

    # cardinality of the group (including inversion symmetry if present!)
    def __len__(self):
        return len(self.S) * (2 if self.inversion else 1)

    # initialize the symmetry group from the lattice matrix A and the basis atoms b
    # b is a list of lists of basis positions with the meaning bpos = b[type][index]
    # TODO unfinished
    def from_lattice(A, b):
        assert False
        # simplify the problem using qr and length normalizsation
        _, A = np.linalg.qr(A)
        A /= A[0,0]
        if len(A) == 2:
            S = [np.eye(2)]
            # check which rotation symmetry is given by just testing them
            # for that, generate the full pointcloud in a circle
            points = (A @ b.T).T
            r = np.max(np.linalg.norm(points, axis=-1))
            # TODO generate a full circle of these points
            for x in [4, 3]:
                angle = 2*np.pi/x
                rot = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
                if pointcloud_distance(rot @ points, points) < 1e-7:
                    # add symmetry
                    S.append(rot)
                    if x == 3:
                        S.append(rot.T)
                    break
            # check if inversion symmetry is given
            # inversion is given, if the basis is equivalent to 1 - basis
            inv = True
            for b_ in b:
                if not pointcloud_distance(b_ % 1.0, (1 - b_) % 1.0) < 1e-7:
                    inv = False
            return Symmetry(S, inversion=inv)
        elif len(A) == 3:
            S = [np.eye(3)]
            # collect symmetries
            # TODO
            # check if inversion symmetry is given
            inv = pointcloud_distance(b % 1.0, (1 - b) % 1.0) < 1e-7
            return Symmetry(S, inversion=inv)
        else:
            raise NotImplementedError(f"not implemented for dimension {len(A)}")

    # apply a basis transformation to all symmetry operations.
    # S' = inv(B) @ S @ B
    def transform(self, basis_transform):
        self.S = np.einsum("nij,mi,jk->nmk", self.S, np.linalg.inv(basis_transform), basis_transform)
        return self

    # one dimensional symmetry (inversion or nothing)
    def one_dim(inversion):
        S = [np.array(((1,),))]
        return Symmetry(S, inversion)
    
    # two dimensional symmetry (n-fold rotation symmetry (includes inversion) where n in {1, 2, 3, 4, 6})
    def two_dim(count):
        assert count in {1, 2, 3, 4, 6}
        inversion = False
        if count % 2 == 0:
            inversion = True
            count /= 2
        if count == 2:
            S = [np.eye(2), np.array(((0,1), (-1,0)))]
        elif count == 3:
            c = np.cos(2*np.pi/3)
            s = np.sin(2*np.pi/3)
            S = [np.eye(2), np.array(((c,s), (-s,c))), np.array(((c,-s), (s,c)))]
        else:
            S = [np.eye(2)]
        return Symmetry(S, inversion)
    
    # create the symmetry group from a set of unique generators.
    def from_generator(G, inversion):
        N = len(G[0])
        G = np.array(G) + 0.0
        assert len(G) > 0, "Need at least one generator (e.g. the neutral element) to determine the dimension"
        d = len(G[0])
        # remove the neutral element
        G = [s for s in G if np.linalg.norm(s - np.eye(d)) > 1e-7]
        if len(G) == 0:
            return Symmetry(np.eye(d)[None,...], inversion)
        if [s for s in G if abs(abs(np.linalg.det(s)) - 1) > 1e-10]:
            raise ValueError("invalid matrix in parameter. All matrices need to have |det| = 1, otherwise the group is infinite")
        # TODO check uniqueness of G
        # add the neutral element back in
        S = [np.eye(d)] + G
        # exponentiate the group n times to form the full group
        for _ in range(1000):
            prev_len = len(S)
            S_new = np.array(S)
            S_new = np.reshape(S_new.reshape(-1, 1, N, N) @ S_new.reshape(1, -1, N, N), (-1, N, N))
            # find unique with a margin of error, assuming the S where unique before
            for s in S_new:
                is_new = True
                for s2 in S:
                    if np.linalg.norm(s - s2) < 1e-7:
                        is_new = False
                        break
                if is_new:
                    S.append(s)
            assert len(S) < 1000, "group size limitation to avoid endless loops"
            if len(S) <= prev_len:
                break
        return Symmetry(np.array(S), inversion)

    # no symmetry
    def none(dim=3):
        return Symmetry([np.eye(dim)], False)
    
    # inversion symmetry
    def inv(dim=3):
        return Symmetry([np.eye(dim)], True)

    # octahedral group https://en.wikipedia.org/wiki/Octahedral_symmetry
    def cubic(inversion):
        return Symmetry.even_perm3() * Symmetry.mirror3(inversion)
    
    # permutation symmetry in the 3 axis
    def perm3(inversion=False):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((0,1,0), (0,0,1), (1,0,0)),
             ((0,0,1), (1,0,0), (0,1,0)),
             ((1,0,0), (0,0,1), (0,1,0)),
             ((0,1,0), (1,0,0), (0,0,1)),
             ((0,0,1), (0,1,0), (1,0,0))]
        return Symmetry(S, inversion=inversion)
    
    # group generated by [[0,0,1], [0,-1,0], [1,0,0]], [[-1,0,0], [0,0,1], [0,1,0]]
    def even_perm3(inversion=False):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((0,1,0), (0,0,-1), (-1,0,0)),
             ((0,0,-1), (1,0,0), (0,-1,0)),
             ((-1,0,0), (0,0,1), (0,1,0)),
             ((0,-1,0), (-1,0,0), (0,0,-1)),
             ((0,0,1), (0,-1,0), (1,0,0))]
        return Symmetry(S, inversion=inversion)

    # point reflections in all 3 planes (Klein four group V_4), or mirror symmetries for all axes if inversion = True
    def mirror3(inversion=False):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((1,0,0), (0,-1,0), (0,0,-1)),
             ((-1,0,0), (0,-1,0), (0,0,1)),
             ((-1,0,0), (0,1,0), (0,0,-1))]
        return Symmetry(S, inversion=inversion)
    
    # mirror symmetry along x axis
    def mirror_x(inversion=False):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((-1,0,0), (0,1,0), (0,0,1))]
        return Symmetry(S, inversion=inversion)

    # symmetries of a 2D square
    def square():
        S = [((0,1), (-1,0)),
             ((1,0), (0,-1))]
        return Symmetry.from_generator(S, False)
    
    # 2D rotation symmetry
    def o2():
        S = [((1,0), (0,1)),
             ((0,1), (-1,0)),
             ((-1,0), (0,-1)),
             ((0,-1), (1,0))]
        return Symmetry(S, False)
    
    def monoclinic_x(inversion):
        # monoclinic crystal (inversion symmetry + 180° rotation in yz)
        D = [np.eye(3), np.diag((1, -1, -1))]
        return Symmetry(D, inversion)
    
    def monoclinic_y(inversion):
        # monoclinic crystal (inversion symmetry + 180° rotation in xz)
        D = [np.eye(3), np.diag((-1, 1, -1))]
        return Symmetry(D, inversion)
    
    def monoclinic_z(inversion):
        # monoclinic crystal (inversion symmetry + 180° rotation in yz)
        D = [np.eye(3), np.diag((-1, -1, 1))]
        return Symmetry(D, inversion)
    
    # check if space dependent function satisfies the symmetry
    # foo is a function k -> matrix
    def check_symmetry(self, foo):
        # random sample points
        r_smpl = np.random.random((50, len(self.S[0])))
        for r in r_smpl:
            values = []
            for s in self.S:
                values.append(foo(s @ r))
            if np.linalg.norm(np.std(values, axis=0)) > 1e-7:
                print("symmetry error")
                print(np.std(values, axis=0))
                return False
        return True
    
    # fill in symmetric k-points in symmetry reduced points.
    # -> creates a ordered grid for cubic symmetry
    # NOTE this method ONLY works for symmetry reduced points, otherwise it will create duplicates
    # returns a list of k-points and an index list for how to get the k-points.
    def realize_symmetric(self, k_smpl):
        order = list(range(len(k_smpl)))
        full_k = list(k_smpl)
        for i, k in enumerate(k_smpl):
            # asymtotically slow* algorithm to find all symmetric points
            # (fine because it's never going above 48, so asymtotic behaviour is irrelevant)
            scale = np.linalg.norm(k)
            if scale == 0:
                continue
            used_k = [k]
            for inv in [1, -1] if self.inversion else [1]:
                for s in self.S:
                    k_ = s @ k * inv
                    # * this is why this algorithm is asymtotically slow
                    if np.min(np.linalg.norm(used_k - k_.reshape(1, -1), axis=-1)) < scale * 1e-6:
                        continue
                    used_k.append(k_)
                    full_k.append(k_)
                    order.append(i)
        # sort by x, y, z compatible with reshape to meshgrid
        order = np.array(order)
        full_k = np.array(full_k)
        for i in range(len(k_smpl[0])):
            # the round here is annoying as it can break at wrong places
            # + np.pi makes it less likely, but it can still happen
            reorder = np.argsort(np.round(full_k[:,i] + np.pi, 4), kind='stable')
            full_k = full_k[reorder]
            order = order[reorder]
        return np.array(full_k), order
    
    # fill symmetry reduced data up to the full dataset
    # -> creates a grid for cubic symmetry
    def realize_symmetric_data(self, k_smpl, reduced):
        full_k, order = self.realize_symmetric(k_smpl)
        return full_k, np.array(np.asarray(reduced)[order])
    
    # reduce symmetric data to only included representants for each k equivalence class
    # NOTE: the result can be unstable but usually it's good
    def reduce_symmetric_data(self, k_smpl, full, checked=False):
        # same sort as in realize_symmetric_data
        k_smpl = np.asarray(k_smpl)
        full = np.asarray(full)
        for i in range(len(k_smpl[0])):
            order = np.argsort(np.round(k_smpl[:,i] + np.pi, 4), kind='stable')
            k_smpl = k_smpl[order]
            full = full[order]
        reduced_k = np.asarray(k_smpl)
        reduced = np.asarray(full)
        # sort by length of k, since all symmetry operations keep length equal
        order = np.argsort(np.linalg.norm(reduced_k, axis=-1)**2, kind='stable') # this stable works really well
        reduced_k = list(reduced_k[order])
        reduced = list(reduced[order])
        i = 0
        while i < len(reduced_k):
            # good* algorithm to find all symmetric points
            k = reduced_k[i]
            value = reduced[i]
            i += 1
            scale = np.linalg.norm(k)
            if scale == 0:
                continue
            for inv in [1, -1] if self.inversion else [1]:
                for s in self.S:
                    k_ = s @ k * inv
                    if np.linalg.norm(k - k_) < scale * 1e-6:
                        continue
                    # find all instances of k_ in reduced_k and delete them
                    j = i
                    while j < len(reduced_k):
                        if np.linalg.norm(reduced_k[j]) > scale * 1.0001:
                            break # * this is what makes this a good algorithm
                        if np.linalg.norm(reduced_k[j] - k_) < scale * 1e-6:
                            if checked and np.linalg.norm(reduced[j] - value) > 1e-7:
                                raise ValueError("Symmetry check found asymmetric data.")
                            del reduced[j]
                            del reduced_k[j]
                        else:
                            j += 1
        return np.array(reduced_k), np.array(reduced)
    
    def complete_neighbors(self, neighbors):
        neighbors, _ = self.realize_symmetric(neighbors)
        neighbors = [n for n in neighbors if next((x for x in n if abs(x) > 1e-7), 1) > 0]
        neighbors = sorted(neighbors, key=lambda n: np.linalg.norm(n))
        return neighbors

    def check_neighbors(self, neighbors):
        assert self.dim() == 3
        # only half of the neighbor terms are present, but the symmetry is important
        neighbors = np.asarray(neighbors)
        neighbors = neighbors[np.argsort(np.linalg.norm(neighbors, axis=-1))]
        count = [0]*len(neighbors)
        for i, r in enumerate(neighbors):
            for s in zip(self.S):
                r_ = s @ r
                # here only the neighbors with an index similar to i need to be checked
                # TODO replace this search by the KD-Trees
                start = max(i - len(self.S) * 3, 0)
                end = min(i + len(self.S) * 3, len(neighbors))
                j = start + np.argmin(np.minimum(np.linalg.norm(neighbors[start:end] - r_[None, :], axis=-1),
                                            np.linalg.norm(neighbors[start:end] + r_[None, :], axis=-1)))
                count[j] += 1
        for c in count:
            if c != len(self.S):
                raise ValueError("neighbors need to be choosen to fit the symmetry, however counting occurences has found the numbers " + str(count))
    
    # calculate the weight of a k point (percent of the space angle around the point) in a 1-periodic lattice with this symmetry
    def k_weight(self, k_smpl):
        weights = np.zeros(len(k_smpl), dtype=np.int32)
        def pingpong_distance(x):
            return 0.5 - np.abs(x % 1.0 - 0.5)
        # NOTE this could be done faster using the right set of generators
        for sign in [-1, 1] if self.inversion else [1]:
            for s in self.S:
                weights += pingpong_distance(np.einsum("ij,nj->ni", sign*s - np.eye(len(s)), k_smpl)).sum(-1) < 1e-7
        return 1 / weights
    
    # calculate the number of unique symmetric points from a given representant.
    # Same as k_weight, but without periodicity
    def r_class_size(self, k_smpl):
        # instead of generating the symmetric points, check how many symmetries fail to produce new points.
        # NOTE every symmetric point can be generated from using the application of just one symmetry operation,
        # therefore the number of points is the number of symmetries divided by the symmetries which leave the initial point invariant.
        weights = np.zeros(len(k_smpl), dtype=np.int32)
        for sign in [-1, 1] if self.inversion else [1]:
            for s in self.S:
                weights += np.linalg.norm(np.einsum("ij,nj->ni", sign*s - np.eye(len(s)), k_smpl), axis=-1) < 1e-7
        return len(self) / weights
    
    def find_classes(self, points):
        points = np.asarray(points)
        classes = {} # {representative_index: { index }}
        covered = set()
        # build the symmetry graph for the neighbor list
        for i, r in enumerate(points):
            rep_class = set()
            for s in self.S:
                r_ = s @ r
                dist = np.linalg.norm(points - r_[None, :], axis=-1)
                j = np.argmin(dist)
                if dist[j] <= 1e-7:
                    rep_class.add(j)
            if i not in covered:
                classes[i] = rep_class
            covered = covered.union(rep_class)
        return classes

    # symmetrize a tensor accoding to this symmetry. This is a projection.
    def symmetrize(self, tensor):
        orig = np.array(tensor) * 1.0
        res = np.zeros_like(orig)
        for s in self.S:
            res += s.T @ orig @ s
        res /= len(self.S)
        return res
    
    # compute equivalence classes with a given equivalence relation.
    # This realizes the inversion symmetry for the result
    def equivalence_classes(self, equiv_relation):
        rem = list(self.S)
        if self.inversion:
            rem = rem + list(-self.S)
        classes = []
        while rem:
            s = rem.pop()
            # find equivalent elements
            unique = [s]
            for i in range(len(rem)-1, -1, -1):
                s2 = rem[i]
                if equiv_relation(s, s2):
                    unique.append(s2)
                    del rem[i]
            classes.append(unique)
        return classes
    
    def conjugacy_classes(self):
        def conjugated(a, b):
            cc = np.einsum("nij,jm,nmk->nik", self.S, a, np.linalg.inv(self.S))
            for c in cc:
                if np.linalg.norm(c - b) < 1e-7:
                    return True
            return False
        return self.equivalence_classes(conjugated)

    # combine two symmetries by finding the smallest symmetry group, that is generated by them
    def __mul__(self, other):
        assert self.dim() == other.dim()
        return Symmetry.from_generator(list(self.S) + list(other.S), inversion=(self.inversion or other.inversion))

    # find the factor group or raise a ValueError
    def __truediv__(self, rhs):
        assert self.dim() == rhs.dim()
        d = self.dim()
        assert (self.inversion or not rhs.inversion) and len(self.S) % len(rhs.S) == 0, "The righthand side needs to be a subgroup"
        # TODO check subgroup
        def left(a, b):
            # to check a*rhs=b*rhs, check a^{-1}*b in rhs
            c = np.linalg.inv(a) @ b
            for r in rhs.S:
                if np.linalg.norm(r - c) < 1e-7:
                    return True
            return False
        classes = self.equivalence_classes(left)
        assert len(classes) == len(self) // len(rhs), "The number of left equivalence classes doesn't match the assumption of a normal subgroup"
        # TODO check the classes for the group properties (if they have it, rhs is a normal subgroup)
        # reduce the classes, while keeping the group property.
        # 1. find the class with the identity and reduce it to just that
        # 2. for each class, find the order and remove all elements,
        #    which do not result in the identity if taken to that power
        while True:
            for i in range(len(classes)):
                # takes increasing powers for each element of the class until one reaches the identity.
                # if more than one element reaches the identity at the same time, keep all of them in this step.
                base = np.array(classes[i])
                powers = np.zeros_like(base) + np.eye(d)[None,:,:]
                for _ in range(len(self.S)+1): # no order bigger than the group size
                    powers = powers @ base
                    classes[i] = np.array([a for j, a in enumerate(base) if np.linalg.norm(powers[j] - np.eye(d)) < 1e-7])
                    if len(classes[i]) > 0:
                        break
            # the class with the identity now has size 1, all the others can have a different size
            # -> take a class with size > 1, select an element and multiply it onto all classes, then repeat
            # -> this creates a second class of size 1
            # -> all classes will have size 1 in the end and obey the group structure
            big_classes = [c for c in classes if len(c) > 1]
            if len(big_classes) == 0:
                break
            #else:
            #    print("rerun with", len(big_classes), "big classes remaining")
            selected = np.linalg.inv(big_classes[0][0]) # guarantees that this will not be an endless loop
            new_classes = []
            for c in classes:
                c_unique = []
                for s in c:
                    s = selected @ s
                    is_new = True
                    for u in c_unique:
                        if np.linalg.norm(u - s) < 1e-7:
                            is_new = False
                            break
                    if is_new:
                        c_unique.append(s)
                new_classes.append(c_unique)
            classes = new_classes
        id_inv = np.linalg.inv(classes[0][0]) # element to permute the identity to the first element
        classes = np.array([c[0] @ id_inv for c in classes])
        sym = Symmetry(classes, inversion=self.inversion and rhs.inversion)
        if not sym.check():
            raise ValueError("The righthand side is no normal subgroup")
        return sym

def test_symmetry():
    assert pointcloud_distance([(1, 2), (3, 4)], [(1, 2), (1, 2)]) > 2
    assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 3)]) == 1
    assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 2)]) == 0
    # TODO add automated tests for realize_symmetric_data, reduce_symmetric_data

    # smallest set of generators for O
    S = [[[0,1,0], [-1,0,0], [0,0,1]],
        [[0,0,1], [1,0,0], [0,1,0]]]
    O = Symmetry.from_generator(S, False)
    assert len(O) == 24
    assert Symmetry.cubic(False) == O
    # generators for a klein four subgroup V_4
    S = [[[-1,0,0], [0,-1,0], [0,0,1]],
        [[1,0,0], [0,-1,0], [0,0,-1]]]
    V_4 = Symmetry.from_generator(S, False)
    assert len(V_4) == 4
    # generators for the factor group ig
    S = [[[0,0,1], [0,-1,0], [1,0,0]],
        [[-1,0,0], [0,0,1], [0,1,0]]]
    F = Symmetry.from_generator(S, False)
    assert len(F) == 6
    assert Symmetry.even_perm3(False) == F
    assert V_4 * F == O
    assert O / V_4 == F
    assert O / Symmetry.mirror3(False) == F
    # 2D rotation
    S = [((0, 1), (-1, 0))]
    R = Symmetry.from_generator(S, False)
    assert Symmetry.o2() == R
    assert len(Symmetry.square() / Symmetry.o2()) == 2

    # test realize_symmetric
    neighbors = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)) # works well
    full_neighbors, order = Symmetry.cubic(True).realize_symmetric(neighbors)
    assert np.linalg.norm(np.linalg.norm(full_neighbors, axis=-1) - np.linalg.norm(np.asarray(neighbors)[order], axis=-1)) < 1e-7
    assert len(order) == 27, f"the symmetrization has yielded {len(order)} elements"
    full_neighbors = Symmetry.cubic(True).complete_neighbors(neighbors)
    assert len(full_neighbors) == (27+1)//2, f"the symmetrization has yielded {len(full_neighbors)} elements"

