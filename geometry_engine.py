import numpy as np

class GeometryEngine:
    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates the angle between three points (a, b, c).
        b is the vertex (e.g. Elbow)
        
        Args:
            a, b, c: Lists or tuples (x, y) or [x, y, z]
        Returns:
            angle: Float in degrees (0 to 180)
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        cosine_angle = dot_product / (norm_ba * norm_bc)

        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) #computers can make rounding errors so using clip

        return np.degrees(angle_rad)
