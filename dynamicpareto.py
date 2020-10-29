#!/usr/bin/env python3

import numpy as np

class DynamicPareto():

    def __init__(self, d, P0=None, maximize=None, trackLabels=False, labels=None):
        """
        d: Number of dimensions in the variable
        P0: if provided, should be an array-like with data points in rows,
            and number of columns equal to d
        maximize: array-like of bools (or 0/1s) indicating whether to
            maximize each variable (minimizes them if False)
        labels: intial list of labels to uniquely identify points in P0
        """
        self.d = d

        # Multiplier is d-array of +/- 1 to toggle between maximizing and
        # minimizing the respective variable.
        if maximize is None:
            self.multiplier = np.ones((d,), dtype=np.float)
        else:
            self.multiplier = np.array(maximize, dtype=np.float) * 2 - 1
            assert np.allclose(np.absolute(self.multiplier), 1), \
                "maximize should be array-like of bool (or 0/1)."

        # If P0 is provided, initialize the Pareto front with it
        # Otherwise, intialize with an np.inf point that will be removed
        # from the front when the first test point is added.
        if P0 is None:
            self.P = np.ones((1,d)) * -np.inf * self.multiplier
        else:
            self.P = np.array(P0)
            assert self.P.ndim == 2, "P0 must be 2d array-like."
            assert self.P.shape[1] == d, "P0 must have d (%d) columns."

        # Determine whether user wants to track labels (so that points on
        # the pareto front can be uniquely identified, not just by the
        # variable values being optimized). If so, track labels.
        # Note: Code does NOT ensure labels are unique!!!!!!!
        if trackLabels:
            self.trackLabels = True
            if P0 is None:
                self.labels = []
            else:
                self.labels = list(labels)
                assert len(self.labels) == self.P.shape[0], "Please provide number of labels equal to number of rows in P0."
        else:
            self.trackLabels = False
            self.labels = None

    def update_pareto(self, x, label=None):
        """
        Tests if test point, x, is on the Pareto front given the current
        points in P. If x is on the new front, updates P to include x, and
        also removes any points previously in P that are no longer on the
        front due to the addition of x. Otherwise, P remains unchanged.
        Returns True if x is on the new Pareto front, and False otherwise.
        Note: the test point, x, is NOT on the Pareto front if you can find
            at least one point in the current P that is larger than x in
            all d dimensions.
        """
        PDiff = (self.P - x) * self.multiplier
        try:
            isPareto = ~np.any(np.all(PDiff >= 0.0, axis=1))
        except ValueError as e:
            print(e)
            print(PDiff)
            print(PDiff > 0.0)
            print(np.all(PDiff > 0.0, axis=1))
            raise

        # If x is on the front, add it to P, and remove any points that are
        # now no longer part of the front
        if isPareto:
            maskNotOnFront = np.all(PDiff <= 0.0, axis=1)
            try: # vstack throws ValueError x is the only point in updated P
                self.P = np.vstack((self.P[~maskNotOnFront,:], np.atleast_2d(x)))
            except ValueError:
                self.P = np.atleast_2d(x)
            if self.trackLabels:
                try: # vstack throws ValueError x is the only point in updated P
                    self.labels = [l for l, m in zip(self.labels, maskNotOnFront) if not m] + [label]
                except ValueError:
                    self.labels = [label]
        return isPareto

    def calculate_average_pareto(self):
        return list(np.mean(self.P, axis=0))

    def calculate_pareto_length(self):
        iSort = np.argsort(self.P[:,0])
        P = self.P[iSort,:]
        d = 0.0
        for i in range(1,len(P)):
            p1, p2 = P[i-1], P[i]
            d += np.sqrt(sum((p1-p2)**2))
        return d


def main():

    import matplotlib.pyplot as plt

    np.random.seed(4242)

    # Generate gaussian distributed values for test
    x1 = np.random.normal(10, 1, 1000)
    x2 = np.random.normal(15, 2, 1000)
    # Plot them in blue
    plt.plot(x1, x2, 'b.')

    # Initialize pareto: 2 variables, maximise x1 and minimize x2
    p = DynamicPareto(d=2, P0=None, maximize=[1,0], trackLabels=True, labels=None)

    # Loop over all pairs of points in x1,x2 and update the pareto front
    # The index i is used as label
    # The length and average of the pareto front are saved for checking the convergence
    plength = list()
    paverage = list()
    for i,d in enumerate(zip(x1, x2)):
        p.update_pareto(d, i)
        plength.append(p.calculate_pareto_length())
        paverage.append(p.calculate_average_pareto())
    paverage = np.array(paverage).transpose()

    # Sort the point on the front
    iSort = np.argsort(p.P[:,0])
    P = p.P[iSort,:]
    labels = np.array(p.labels)[iSort]

    # Print and plot them
    print('Points on the Pareto front:')
    for d, l in zip(P,labels):
        plt.plot(d[0], d[1], 'r.')
        plt.annotate('%d' % l, xy=d, textcoords='data')
        print(l, d)

    print('Pareto average :', p.calculate_average_pareto())
    print('Pareto length  :', p.calculate_pareto_length())

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    plt.plot(plength, label='Length')
    plt.plot(paverage[0], label='Avg x1')
    plt.plot(paverage[1], label='Avg x2')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
 
