import numpy as np
import os
import random
from scipy import ndimage
import torch

import numpy as np
import os
import random
from scipy import ndimage
import torch


class PhantomGenerator:
    def __init__(self, seed=1029, image_size=128):
        self.seed = seed
        self.image_size = image_size
        self._seed_all()

    def _seed_all(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def generate_circle(self, n_samples=1000):
        scale_range = np.random.uniform(1.8, 2.2, n_samples)
        shift_x_range = np.random.uniform(1.5, 3, n_samples)
        shift_y_range = np.random.uniform(1.5, 3, n_samples)

        def mask(h, w, scale, shift_x, shift_y):
            center = (int(w / shift_x), int(h / shift_y))
            radius = min(center[0], center[1], w - center[0], h - center[1])
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            return (dist <= radius / scale).astype(int)

        return [mask(self.image_size, self.image_size, scale_range[i], shift_x_range[i], shift_y_range[i])
                for i in range(n_samples)]

    def generate_ellipse(self, n_samples=3000):
        scale_range = np.random.uniform(0.8, 1.2, n_samples)
        w_range = np.random.randint(-15, 15, n_samples)
        h_range = np.random.randint(-15, 15, n_samples)
        rotation_range = np.linspace(0, 180, 36, False)
        rotation_label = np.random.randint(0, 36, n_samples)

        def mask(scale, h_shift, w_shift):
            center = (self.image_size // 2 + w_shift, self.image_size // 2 + h_shift)
            Y, X = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt(((X - center[0]) / 18) ** 2 + ((Y - center[1]) / 35) ** 2)
            return (dist <= scale).astype(int)

        data = []
        for i in range(n_samples):
            ph = mask(scale_range[i], h_range[i], w_range[i])
            ph = ndimage.rotate(ph, rotation_range[rotation_label[i]], reshape=False)
            ph = (ph >= 0.5).astype(int)
            data.append(ph)
        return data

    def generate_triangle(self, n_samples=3000):
        v_a = np.random.uniform(-10, 2.5, n_samples)
        rotation_range = np.linspace(0, 180, 36, False)
        rotation_label = np.random.randint(0, 36, n_samples)

        def mask(a_range):
            size = self.image_size
            size1 = int(size + 5 * a_range)
            xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
            x = round(size1 / 2)
            y = round(size1 / 2)
            slope = 0
            z1 = (yy - 3/2*y < -slope*(xx - slope*3/2*x/(1 + slope**2)**0.5))
            z2 = (slope*(yy - slope*3/2*y/(1 + slope**2)**0.5) < (xx - 1/2*x))
            z3 = (yy - 1/2*y > -slope*(xx + slope*(-x)/(1 + slope**2)**0.5))
            z4 = (slope*(yy - slope*y/(1 + slope**2)**0.5) > (xx - 3/2*x))
            z5 = (yy - y + 1 > -(1 - slope)*(xx - x))
            P = z1 * z2 * z3 * z4 * z5
            return P.astype(int)

        data = []
        for i in range(n_samples):
            ph = mask(v_a[i])
            ph = ndimage.rotate(ph, rotation_range[rotation_label[i]], reshape=False)
            ph = (ph >= 0.5).astype(int)
            data.append(ph)
        return data

    def generate_mixed(self, n_samples=3000):
        def create_triangle(mx, my):
            Y, X = np.ogrid[:128, :128]
            m1 = X - mx < 100 + my
            m2 = -X + 120 / np.tan(np.pi / 4) + mx < Y / np.tan(np.pi / 4) + my
            m3 = Y < 100 + my
            return (m1 & m2 & m3).astype(int)

        def create_pentagon(mx, my):
            Y, X = np.ogrid[:128, :128]
            t36 = np.tan(36 * np.pi / 180)
            t72 = np.tan(72 * np.pi / 180)
            m1 = (X - 30) - 10 / t36 - mx < Y / t36 - my
            m2 = -(X + 30) + 83 / t36 + mx < Y / t36 - my
            m3 = -(X + 30) + 477 / t72 + mx > Y / t72 + my
            m4 = (X - 30) + 83 / t72 - mx > Y / t72 + my
            m5 = Y + my < 100
            return (m1 & m2 & m3 & m4 & m5).astype(int)

        def create_hexagon(mx, my):
            Y, X = np.ogrid[:128, :128]
            t30 = np.tan(30 * np.pi / 180)
            m1 = (X - 30) - 10 / t30 - mx < Y / t30 - my
            m2 = -(X + 30) + 64 / t30 + mx < Y / t30 - my
            m3 = (X - 30) - 10 / t30 - mx > Y / t30 + my - 180
            m4 = -(X + 30) + 64 / t30 + mx > Y / t30 + my - 180
            m5 = X > 20 + mx + my / 2
            m6 = X < 108 + mx - my / 2
            return (m1 & m2 & m3 & m4 & m5 & m6).astype(int)

        def gen_rotated(shape_func, mx_range, my_range, rotation_range, label_range):
            out = []
            for i in range(n_samples):
                ph = shape_func(mx_range[i], my_range[i])
                ph = ndimage.rotate(ph, rotation_range[label_range[i]], reshape=False)
                ph = (ph >= 0.5).astype(int)
                out.append(ph)
            return out

        rot_range = np.linspace(0, 180, 36, False)

        triangle = gen_rotated(
            create_triangle,
            np.random.randint(-10, 10, n_samples),
            np.random.randint(-10, 0, n_samples),
            rot_range,
            np.random.randint(0, 36, n_samples)
        )

        pentagon = gen_rotated(
            create_pentagon,
            np.random.randint(-10, 10, n_samples),
            np.random.randint(0, 10, n_samples),
            rot_range,
            np.random.randint(0, 36, n_samples)
        )

        hexagon = gen_rotated(
            create_hexagon,
            np.random.randint(0, 10, n_samples),
            np.random.randint(0, 10, n_samples),
            rot_range,
            np.random.randint(0, 36, n_samples)
        )

        P_all = triangle + pentagon + hexagon
        random.shuffle(P_all)
        return P_all

    def generate_mixed_pentagon(self, n_samples=3000):
        def create_pentagon(mx, my):
            Y, X = np.ogrid[:128, :128]
            t36 = np.tan(36 * np.pi / 180)
            t72 = np.tan(72 * np.pi / 180)
            m1 = (X - 30) - 10 / t36 - mx < Y / t36 - my
            m2 = -(X + 30) + 83 / t36 + mx < Y / t36 - my
            m3 = -(X + 30) + 477 / t72 + mx > Y / t72 + my
            m4 = (X - 30) + 83 / t72 - mx > Y / t72 + my
            m5 = Y + my < 100
            return (m1 & m2 & m3 & m4 & m5).astype(int)

        return self._rotate_batch(create_pentagon, n_samples, (-10, 10), (0, 10))

    def generate_mixed_hexagon(self, n_samples=3000):
        def create_hexagon(mx, my):
            Y, X = np.ogrid[:128, :128]
            t30 = np.tan(30 * np.pi / 180)
            m1 = (X - 30) - 10 / t30 - mx < Y / t30 - my
            m2 = -(X + 30) + 64 / t30 + mx < Y / t30 - my
            m3 = (X - 30) - 10 / t30 - mx > Y / t30 + my - 180
            m4 = -(X + 30) + 64 / t30 + mx > Y / t30 + my - 180
            m5 = X > 20 + mx + my / 2
            m6 = X < 108 + mx - my / 2
            return (m1 & m2 & m3 & m4 & m5 & m6).astype(int)

        return self._rotate_batch(create_hexagon, n_samples, (0, 10), (0, 10))

    def _rotate_batch(self, shape_func, n_samples, mx_range, my_range):
        mx = np.random.randint(*mx_range, n_samples)
        my = np.random.randint(*my_range, n_samples)
        rot_range = np.linspace(0, 180, 36, False)
        rot_labels = np.random.randint(0, 36, n_samples)

        data = []
        for i in range(n_samples):
            ph = shape_func(mx[i], my[i])
            ph = ndimage.rotate(ph, rot_range[rot_labels[i]], reshape=False)
            ph = (ph >= 0.5).astype(int)
            data.append(ph)
        return data

    def generate_shepp_logan_fix(self, n_samples=3000):
        r2_range = np.linspace(0, 180, 36, endpoint=False)
        r2_label = np.random.randint(0, len(r2_range), n_samples)
        ma2_range = np.random.uniform(0.11, 0.13, n_samples)
        mi2_range = np.random.uniform(0.30, 0.32, n_samples)

        def shepp_logan_single(r2, ma2, mi2):
            def ellipse_mask(X, Y, xc, yc, a, b, theta):
                ct, st = np.cos(theta), np.sin(theta)
                return (((X - xc) * ct + (Y - yc) * st) ** 2 / a ** 2 +
                        ((X - xc) * st - (Y - yc) * ct) ** 2 / b ** 2) < 1

            X, Y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
            theta1 = 0
            theta2 = np.deg2rad(45)
            theta3 = np.deg2rad(r2)

            roi1 = ellipse_mask(X, Y, 0, 0, 0.54, 0.84, theta1)
            roi2 = ellipse_mask(X, Y, 0.22, 0, 0.10, 0.28, theta2)
            roi3 = ellipse_mask(X, Y, -0.22, 0, ma2, mi2, theta3)

            ph = np.zeros_like(X)
            ph[roi1] = 1
            ph[roi2] = 1
            ph[roi3] = 1
            ph[ph < 0.01] = 0
            return ph

        return [shepp_logan_single(r2_range[r2_label[i]], ma2_range[i], mi2_range[i])
                for i in range(n_samples)]

    def generate_shepp_logan(self, n_samples=3000):
        def ct_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2):
            E = np.zeros((10, 6))
            E[:, 0] = [2, -.98, -.02, -.02, .01, .01, .01, .01, .01, .01]
            E[:, 1] = [.58, .54, ma1, ma2, .21, .046, .046, .046, .023, .023]
            E[:, 2] = [.92, .874, mi1, mi2, .25, .046, .046, .023, .023, .046]
            E[:, 3] = [0, 0, .22, -.22, 0, 0, 0, -.08, 0, .06]
            E[:, 4] = [0, -.0184, 0, 0, .35, .1, -.1, -.605, -.605, -.605]
            E[:, 5] = np.deg2rad([0, 0, r1, r2, 0, 0, 0, 0, 0, 0])
            return E

        def ct_modified_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2):
            E = ct_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2)
            E[:, 0] = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            return E

        def ct_shepp_logan_2d(M, N, r1, r2, ma1, mi1, ma2, mi2):
            E = ct_modified_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2)
            X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, M))
            ph = np.zeros((M, N))
            for A, a, b, xc, yc, theta in E:
                ct, st = np.cos(theta), np.sin(theta)
                mask = (((X - xc) * ct + (Y - yc) * st) ** 2 / a ** 2 +
                        ((X - xc) * st - (Y - yc) * ct) ** 2 / b ** 2) <= 1
                ph[mask] += A
            return ph

        r1_range = np.linspace(0, 180, 36, False)
        r2_range = np.linspace(0, 180, 36, False)
        r1_label = np.random.randint(0, len(r1_range), n_samples)
        r2_label = np.random.randint(0, len(r2_range), n_samples)
        ma2_range = np.random.uniform(0.06, 0.10, n_samples)
        mi2_range = np.random.uniform(0.24, 0.28, n_samples)
        ma3_range = np.random.uniform(0.11, 0.13, n_samples)
        mi3_range = np.random.uniform(0.30, 0.32, n_samples)

        P_all = []
        for i in range(n_samples):
            r1, r2 = r1_range[r1_label[i]], r2_range[r2_label[i]]
            ma2, mi2, ma3, mi3 = ma2_range[i], mi2_range[i], ma3_range[i], mi3_range[i]
            ph = ct_shepp_logan_2d(128, 128, r1, r2, ma2, mi2, ma3, mi3)

            X, Y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
            roi1 = (((X) ** 2) / 0.54 ** 2 + (Y ** 2) / 0.84 ** 2) < 1
            roi2 = (((X - .22) * np.cos(np.deg2rad(r1)) + Y * np.sin(np.deg2rad(r1))) ** 2 / ma2 ** 2 +
                    ((X - .22) * np.sin(np.deg2rad(r1)) - Y * np.cos(np.deg2rad(r1))) ** 2 / mi2 ** 2) < 1
            roi3 = (((X + .22) * np.cos(np.deg2rad(r2)) + Y * np.sin(np.deg2rad(r2))) ** 2 / ma3 ** 2 +
                    ((X + .22) * np.sin(np.deg2rad(r2)) - Y * np.cos(np.deg2rad(r2))) ** 2 / mi3 ** 2) < 1

            ph[~roi1] = 0
            ph[roi2] = 1
            ph[roi3] = 1
            ph[ph < 0.01] = 0

            P_all.append(ph)

        return P_all
