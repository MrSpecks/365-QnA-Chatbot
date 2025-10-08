#!/usr/bin/env python3
"""
Deployment verification script
Run this to check if all files are ready for deployment
"""

import os

def check_deployment_files():
    """Check if all required files exist and have correct content"""
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        '.streamlit/secrets.toml.example'
    ]
    
    print("🔍 Checking deployment files...")
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    # Check streamlit_app.py for correct import
    if os.path.exists('streamlit_app.py'):
        with open('streamlit_app.py', 'r') as f:
            content = f.read()
            if 'from langchain_community.document_loaders.pdf import PyPDFLoader' in content:
                print("✅ Correct PyPDFLoader import found")
            else:
                print("❌ PyPDFLoader import not found or incorrect")
                all_good = False
            
            if '_uploaded_file' in content and '_vectorstore' in content:
                print("✅ Caching parameters correctly prefixed with underscore")
            else:
                print("❌ Caching parameters may need underscore prefix")
                all_good = False
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'langchain-community>=0.3.31' in content:
                print("✅ Updated langchain-community version found")
            else:
                print("❌ langchain-community version may need updating")
                all_good = False
    
    if all_good:
        print("\n🎉 All checks passed! Ready for deployment.")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Fix imports and caching issues'")
        print("3. git push")
        print("4. Redeploy on Streamlit Cloud")
    else:
        print("\n❌ Some issues found. Please fix them before deploying.")
    
    return all_good

if __name__ == "__main__":
    check_deployment_files()
