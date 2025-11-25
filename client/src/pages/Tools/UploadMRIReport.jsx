import { useState, useEffect, useRef } from 'react';
import { assets } from '../../assets/assets';
import { useAppContext } from '../../context/AppContext';
import ButtonLoader from '../../components/ButtonLoader';
import { toast } from 'react-toastify';

const UploadMRIReport = () => {
    const { user, axios, loading, setLoading } = useAppContext();

    const defaultDocument = {
        userId: '',
        name: '',
        images: Array(1).fill(null),
    };

    const [document, setDocument] = useState(defaultDocument);
    const [previewFile, setPreviewFile] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    const imageRefs = useRef([]);

    useEffect(() => {
        if (user?._id) {
            setDocument((prev) => ({ ...prev, userId: user._id }));
        }
    }, [user]);

    const handleChange = (e) => {
        const { id, value } = e.target;
        setDocument((prev) => ({ ...prev, [id]: value }));
    };

    const handleImageChange = (index, file) => {
        const updatedImages = [...document.images];
        updatedImages[index] = file;
        setDocument((prev) => ({ ...prev, images: updatedImages }));
    };

    const removeImage = (index) => {
        const updatedImages = [...document.images];
        updatedImages[index] = null;
        setDocument((prev) => ({ ...prev, images: updatedImages }));

        if (imageRefs.current[index]) {
            imageRefs.current[index].value = '';
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!document.images.some(img => img)) {
            toast.error("Please upload your report.");
            return;
        }

        toast.loading("Processing, Please don't refresh...")

        setLoading(true);
        try {
            const documentData = {
                userId: document.userId,
                name: document.name,
            };

            const formData = new FormData();
            formData.append('documentData', JSON.stringify(documentData));

            document.images.forEach((file) => {
                if (file) formData.append('images', file);
            });

            const { data } = await axios.post('/api/document/add', formData);

            if (data.success) {
                toast.dismiss()
                toast.success(data.message);
                setDocument(defaultDocument);
                imageRefs.current.forEach((ref) => ref && (ref.value = ''));

            } else {
                toast.dismiss()
                toast.error(data.message);
            }
        } catch (error) {
            toast.dismiss()
            toast.error(error.response?.data.message || error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className=" container mx-auto no-scrollbar flex-1 h-[90vh] overflow-y-scroll flex flex-col  items-center">
            <form onSubmit={handleSubmit} className="pt-8 space-y-5 flex justify-center flex-col items-center">
                <div>
                    <h1 className="text-xl md:text-2xl text-special-red font-medium text-center">Upload MRI Report <span className='text-sm text-gray-500'>(png/jpg)</span></h1>
                    <div className="flex flex-wrap items-center justify-center gap-3 mt-8">
                        {document.images?.map((image, index) => (
                            <label key={index} className="relative w-fit">
                                <input
                                    type="file"
                                    id={`image${index}`}
                                    hidden
                                    accept="image/*"
                                    ref={(el) => (imageRefs.current[index] = el)}
                                    disabled={loading}
                                    onChange={(e) => handleImageChange(index, e.target.files[0])}
                                />
                                <div
                                    className="overflow-hidden bg-gray-100 flex items-center justify-center cursor-pointer border border-gray-300 duration-200 transition-all ease-in-out hover:scale-105"
                                    onClick={() => !loading && imageRefs.current[index]?.click()}
                                >
                                    {image ? (
                                        <span className="w-34 h-22 text-2xl text-center text-gray-600 flex flex-col justify-center">
                                            🧠 <p className='text-sm'>Image Uploaded</p>
                                        </span>
                                    ) : (
                                        <img
                                            src={assets.upload_area}
                                            alt="upload"
                                            className="max-w-34"
                                        />
                                    )}
                                </div>
                                {image && (
                                    <button
                                        type="button"
                                        className="absolute -top-2 -right-2 bg-primary text-white rounded-full w-5 h-5 flex items-center justify-center text-xs"
                                        onClick={() => removeImage(index)}
                                        disabled={loading}
                                    >
                                        ✕
                                    </button>
                                )}
                            </label>
                        ))}
                    </div>
                </div>

                <div className="flex flex-col gap-1 w-10 items-center">
                    <input
                        type="text"
                        id="name"
                        value={document.name}
                        onChange={handleChange}
                        className="outline-none md:py-2.5 py-2 px-3 text-center border-b border-gray-500/40 focus:border-special-red capitalize"
                        placeholder="Document Name?"
                        required
                        disabled={loading}
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className={`w-36 px-8 py-2.5 text-white hover:bg-special-red rounded ${loading ? 'bg-special-red/60 cursor-not-allowed' : 'cursor-pointer bg-special-red/90'}`}
                >
                    {loading ? <ButtonLoader /> : 'Upload'}
                </button>
            </form>

            <div className="flex flex-wrap gap-4 my-6">
                {document.images.map((file, idx) =>
                    file ? (
                        <div key={idx} className="border border-gray-300 w-60  rounded overflow-hidden shadow-sm flex flex-col cursor-pointer"
                            onClick={() => {
                                setPreviewFile(file);
                                setIsModalOpen(true);
                            }}
                        >
                            <div className="p-2 bg-gray-100 text-xs text-center truncate">
                                {file.name}
                            </div>
                            <div className="flex-1">
                                {file.type.startsWith('image/') ? (
                                    <img
                                        src={URL.createObjectURL(file)}
                                        alt={`preview-${idx}`}
                                        className="object-contain w-full h-full"
                                    />
                                ) : file.type === 'application/pdf' ? (
                                    <embed
                                        src={URL.createObjectURL(file)}
                                        type="application/pdf"
                                        className="w-full h-full"
                                    />
                                ) : (
                                    <div className="p-4 text-center text-sm text-gray-500">
                                        Preview not available
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : null
                )}
            </div>
            {isModalOpen && previewFile && (
                <div className="fixed inset-0 bg-black bg-opacity-80 z-50 flex items-center justify-center">
                    <div className="relative w-full h-full max-w-5xl max-h-[90vh] p-4">
                        <button
                            onClick={() => setIsModalOpen(false)}
                            className="absolute top-4 right-4 bg-primary text-white p-2 rounded-full z-50"
                        >
                            ✕
                        </button>

                        {previewFile.type.startsWith('image/') ? (
                            <img
                                src={URL.createObjectURL(previewFile)}
                                alt="Full Preview"
                                className="w-full h-full object-contain"
                            />
                        ) : previewFile.type === 'application/pdf' ? (
                            <embed
                                src={URL.createObjectURL(previewFile)}
                                type="application/pdf"
                                className="w-full h-full"
                            />
                        ) : (
                            <div className="text-white text-center mt-10">
                                Preview not supported.
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default UploadMRIReport;

