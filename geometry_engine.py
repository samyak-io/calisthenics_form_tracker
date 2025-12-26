import numpy as np

class GeometryEngine:
    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates the angle between three points (a, b, c).
        b is the vertex (e.g. Elbow)
        
        Args:
            a, b, c: Lists or tuples [x, y] or [x, y, z]
        Returns:
            angle: Float in degrees (0 to 180)
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) #computers can make rounding errors so using clip

        degrees = np.degrees(angle)

        return degrees