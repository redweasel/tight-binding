import numpy as np

# My own symmetries.
# The following has a lot of it, but is missing a part which I deemed crucial for the performance.
# https://qsymm.readthedocs.io/en/latest/tutorial/bloch_generator.html

def pointcloud_distance(pointcloud1, pointcloud2):
    pointcloud1 = np.asarray(pointcloud1)
    pointcloud1 = pointcloud1.reshape(-1, pointcloud1.shape[-1])
    pointcloud2 = np.asarray(pointcloud2).reshape(-1, pointcloud1.shape[-1])
    # for each point in pointcloud1 find the closest in pointcloud2, add the distance, then remove that
    dist = 0.0
    for i, p1 in enumerate(pointcloud1):
        d = np.linalg.norm(p1 - pointcloud2, axis=-1)
        min_index = np.argmin(d.flat)
        dist += d.flat[min_index]
        d = np.delete(d.flat, min_index)
    return dist

# class for symmetries. All symmetries (except inversion symmetry) are saved as unitary/orthogonal transformation matrices
# TODO do they need to be orthogonal? Why not SL(3) or -SL(3)? Those would also create closed sets of symmetries.
class Symmetry:
    # initialize symmetry from orthogonal matrices
    def __init__(self, S, inversion=False):
        self.S = np.asarray(S)
        self.inversion = inversion
        if [s for s in S if abs(abs(np.linalg.det(s)) - 1) > 1e-7]:
            raise ValueError("invalid matrix in symmetry. All matrices need to have |det| = 1")
        # check if S has inversion symmetry build in
        if not inversion and pointcloud_distance(np.reshape(S, (len(S), -1)), -np.reshape(S, (len(S), -1))) < 1e-6:
            print("found inversion symmetry")
            self.inversion = True
            # keep only the S with positive determinants
            self.S = np.array([s for s in S if np.linalg.det(s) > 0])
        # TODO add lists for broken symmetries
        # each symmetry(except for inversion) has a line or a plane on which it's unbroken
        # -> add information about that and add a way to get all the unbroken/broken symmetries for a point
    
    def copy(self):
        return Symmetry(np.array(self.S), self.inversion)

    def dim(self):
        return len(self.S[0])
    
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
    
    # create the symmetry group from a generator set.
    # make sure to always include the neutral element first!
    def from_generator(G, inversion):
        N = len(G[0])
        S = G.copy()
        G = np.array(G, dtype=np.float64)
        assert len(G) > 0, "Need at least one generator (the neutral element)"
        if len(G) == 1:
            assert np.linalg.norm(G[0] @ G[0] - G[0]) < 1e-7, "The first generator needs to be the neutral element"
            return Symmetry(np.array(G, dtype=np.float64), inversion)
        assert np.linalg.norm(G[0] @ G[1] - G[1]) < 1e-7, "The first generator needs to be the neutral element"
        # exponentiate the group n times to form the full group
        for _ in range(1000):
            S_new = np.array(S, dtype=np.float64)
            S_new = np.reshape(S_new.reshape(-1, 1, N, N) @ S_new.reshape(1, -1, N, N), (-1, N, N))
            # convert back to tuples and reduce
            prev_len = len(S)
            S = set()
            for s in S_new:
                S.add(tuple([tuple(s_) for s_ in s]))
            S = list(S)
            assert len(S) < 1000 # limitation to avoid endless loops
            if len(S) <= prev_len:
                break
        return Symmetry(np.array(S, dtype=np.float64), inversion)

    # no symmetry
    def none(dim=3):
        return Symmetry([np.eye(dim)], False)
    
    # inversion symmetry
    def inv(dim=3):
        return Symmetry([np.eye(dim)], True)

    # octahedral group https://en.wikipedia.org/wiki/Octahedral_symmetry
    def cubic(inversion):
        return Symmetry.perm3() * Symmetry.mirror3(inversion)
    
    # TODO symmetry for fcc data from quantum espresso
    def fcc(inversion):
        # compute everything with integer matrices and convert to float at the end
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((1,0,0), (0,0,1), (0,-1,0)),
             ((0,0,-1), (0,1,0), (1,0,0)),
             ((0,1,0), (-1,0,0), (0,0,1))]
        return Symmetry.from_generator(S, inversion)
    
    # permutation symmetry in the 3 axis
    def perm3(inversion=False):
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((0,1,0), (0,0,1), (1,0,0)),
             ((0,0,1), (1,0,0), (0,1,0)),
             ((1,0,0), (0,0,1), (0,1,0)),
             ((0,1,0), (1,0,0), (0,0,1)),
             ((0,0,1), (0,1,0), (1,0,0))]
        return Symmetry(S, inversion=inversion)

    # point reflections in all 3 planes, or mirror symmetries for all axes if inversion = True
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
        # compute everything with integer matrices and convert to float at the end
        # TODO check!
        S = [((1,0), (0,1)),
             ((0,1), (-1,0)),
             ((1,0), (0,-1))]
        return Symmetry.from_generator(S, True)
    
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
    
    # fill symmetry reduced data up to the full dataset
    # -> creates a grid for cubic symmetry and a hexagon for hexagonal symmetry
    def realize_symmetric_data(self, k_smpl, reduced):
        full = list(reduced)
        full_k = list(k_smpl)
        for data, k in zip(reduced, k_smpl):
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
                    full.append(data)
                    full_k.append(k_)
        # sort by x, y, z compatible with reshape to meshgrid
        full_k = np.array(full_k)
        full = np.array(full)
        for i in range(len(k_smpl[0])):
            # the round here is annoying as it can break at wrong places
            # + np.pi makes it less likely, but it can still happen
            order = np.argsort(np.round(full_k[:,i] + np.pi, 4), kind='stable')
            full_k = full_k[order]
            full = full[order]
        return full_k, full
    
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
        neighbors, _ = self.realize_symmetric_data(neighbors, [[0]] * len(neighbors))
        neighbors = [n for n in neighbors if n[0] > 0 or (n[0] == 0 and (n[1] > 0 or (n[1] == 0 and n[2] >= 0)))]
        neighbors = sorted(neighbors, key=lambda n: np.linalg.norm(n))
        return neighbors
    
    def check_neighbors(self, neighbors):
        # only half of the neighbor terms are present, but the symmetry is important
        neighbors = np.asarray(neighbors)
        neighbors = neighbors[np.argsort(np.linalg.norm(neighbors, axis=-1))]
        count = [0]*len(neighbors)
        for i, r in enumerate(neighbors):
            for s in zip(self.S):
                r_ = s @ r
                # here only the neighbors with an index similar to i need to be checked
                # TODO replace these with the correct numbers
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
    
    # combine two symmetries by taking all combinations of multiplications and removing duplicates
    def __mul__(self, other):
        assert self.dim() == other.dim()
        N = self.dim()
        S1 = np.array(self.S, dtype=np.float64)
        S2 = np.array(other.S, dtype=np.float64)
        S = np.reshape(S1[:,None,...] @ S2[None,...], (-1, N, N))
        S = np.unique(S, axis=0)
        return Symmetry(S, inversion=(self.inversion or other.inversion))


# test Symmetry class
#print(np.array(Symmetry(False).S))

# automated tests
assert pointcloud_distance([(1, 2), (3, 4)], [(1, 2), (1, 2)]) > 2
assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 3)]) == 1
assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 2)]) == 0
# TODO add automated tests for realize_symmetric_data, reduce_symmetric_data
