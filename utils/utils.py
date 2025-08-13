class Utils:
    @staticmethod
    def chunks(array: list, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(array), n):
            yield array[i:i + n]