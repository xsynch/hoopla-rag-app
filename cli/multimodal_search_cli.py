import argparse


from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    

    img_parser = subparsers.add_parser("verify_image_embedding", help="Image path and name to use for search")
    img_parser.add_argument("imagepath",type=str,help="Path to the image")

    
    


    args = parser.parse_args()
    
    # run_multimodal_search(args.image,args.query)

    match args.command:
        case "verify_image_embedding":            
            run_get_image_embeddings(args.imagepath)

        case _:
            parser.print_help()

def run_get_image_embeddings(image):
    verify_image_embedding(image)

if __name__ == "__main__":
    main()