from utils import Scrapper, Config


def main(opts):
    # download samples from google
    if opts.download_data:
        scrapper = Scrapper(opts.file_with_classes, opts.path_to_driver)
        scrapper(opts.samples_per_class)


if __name__ == "__main__":
    args = Config(path_to_data='.images/')
    main(args)
