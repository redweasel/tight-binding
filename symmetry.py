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
class Symmetry:
    # initialize symmetry from orthogonal matrices
    def __init__(self, S, inversion=False):
        self.S = S
        self.inversion = inversion
        # check if S has inversion symmetry build in
        if not inversion and pointcloud_distance(np.reshape(S, (len(S), -1)), -np.reshape(S, (len(S), -1))) < 1e-6:
            print("found inversion symmetry")
            self.inversion = True
            # keep only the S with positive determinants
            self.S = [s for s in S if np.linalg.det(s) > 0]
        # TODO add lists for broken symmetries
        # each symmetry(except for inversion) has a line or a plane on which it's unbroken
        # -> add information about that and add a way to get all the unbroken/broken symmetries for a point
    
    def copy(self):
        return Symmetry(np.array(self.S), self.inversion)

    def dim(self):
        return len(self.S[0])

    # initialize the symmetry group from the lattice matrix A and the basis atoms b
    # b is a list of lists of basis positions with the meaning bpos = b[type][index]
    def from_lattice(A, b):
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

    # octahedral group https://en.wikipedia.org/wiki/Octahedral_symmetry
    def cubic(inversion):
        # compute everything with integer matrices and convert to float at the end
        S = [((1,0,0), (0,1,0), (0,0,1)),
             ((1,0,0), (0,0,1), (0,-1,0)),
             ((0,0,-1), (0,1,0), (1,0,0)),
             ((0,1,0), (-1,0,0), (0,0,1))]
        return Symmetry.from_generator(S, inversion)
    
    # symmetries of a 2D square
    def square(inversion):
        # compute everything with integer matrices and convert to float at the end
        S = [((1,0), (0,1)),
             ((0,1), (-1,0))]
        return Symmetry.from_generator(S, inversion)
    
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
    def reduce_symmetric_data(self, k_smpl, full):
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
                            del reduced[j]
                            del reduced_k[j]
                        else:
                            j += 1
        return np.array(reduced_k), np.array(reduced)
    
    # symmetrize a tensor accoding to this symmetry. This is a projection.
    def symmetrize(self, tensor):
        orig = np.array(tensor) * 1.0
        res = np.zeros_like(orig)
        for s in self.S:
            res += s.T @ orig @ s
        res /= len(self.S)
        return res


# test Symmetry class
#print(np.array(Symmetry(False).S))

# automated tests
assert pointcloud_distance([(1, 2), (3, 4)], [(1, 2), (1, 2)]) > 2
assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 3)]) == 1
assert pointcloud_distance([(1, 2), (3, 4)], [(3, 4), (1, 2)]) == 0
# TODO add automated tests for realize_symmetric_data, reduce_symmetric_data
