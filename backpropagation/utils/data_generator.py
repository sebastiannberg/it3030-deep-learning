from utils.doodler_forall import gen_standard_cases


class DataGenerator:

    def generate_dataset(self, count, train_size, validation_size, test_size, n, wr, hr, noise, types, center: bool, flatten: bool):
        if n < 10 or n > 50:
            raise ValueError("Invalid n size in data generator. Should be 10<=n<=50")
        if noise < 0 or noise > 1:
            raise ValueError("Noise should be a fraction")
        if int(round(train_size + validation_size + test_size, 2)) != 1:
            raise ValueError("Sum of train_size, validation_size and test_size must be 1")

        train_size = int(train_size * count)
        validation_size = int(validation_size * count)
        test_size = int(test_size * count)

        train = gen_standard_cases(count=train_size, rows=n, cols=n,
                                   types=types, wr=wr, hr=hr, noise=noise, cent=center, flat=flatten, show=False)
        validation = gen_standard_cases(count=validation_size, rows=n, cols=n,
                                        types=types, wr=wr, hr=hr, noise=noise, cent=center, flat=flatten, show=False)
        test = gen_standard_cases(count=test_size, rows=n, cols=n,
                                  types=types, wr=wr, hr=hr, noise=noise, cent=center, flat=flatten, show=False)

        return train, validation, test
